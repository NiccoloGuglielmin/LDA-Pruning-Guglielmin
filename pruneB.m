%Pruning LDA
%Parametri: rete da fare pruning, rete in output, quantità di importanza da
%           togliere (tot=1), nome del layer fc pre-softmax, metodo usato 
%           per scegliere le importanze iniziali (LDA o LRP o altro), lista
%           di layer o lista di nomi di layer da fare pruning
function [dlnet,loss_pruned,loss] = pruneB(net,tX,tY,A,TRESHOLD,PREDECISION_FC,INITIAL_IMPORTANCE,LAYERS_TO_PRUNE,PLOT_OPTIONAL)

    %Caricamento rete e dataset per il testing
    load('NetAndData.mat');
    numClasses = 10;
    testImages = dlarray(cast(testImages,'single'),'SSCB');
    yy = (yy' == 1:numClasses)'.*1;

    %Conversione della dlnet in un lgraph per consentire la modifica
    lgraph = layerGraph(net.Layers(1:length(net.Layers)));
    %figure
    %plot(lgraph)

    %Calcolo delle importanze per il layer pre-decisionale (Relevances), 
    %calcolate con LDA (come suggerito dal paper) o con LRP (l'importanza
    %è calcolata in base ai pesi dei collegamenti dell'ultimo layer 
    %fullyconnected)
    if strcmp(INITIAL_IMPORTANCE,'LDA')

        %tX: input dell'ultimo fc layer
        %tY: classi corrispondenti per i vari input

        %Rimozione delle dimensioni con varianza quasi nulla
        V = var(tX); %Restituisce la varianza degli elementi di A lungo la 
                     %prima dimensione dell'array la cui dimensione non è uguale a 1.
        I = find(V<1e-10); %Si trovano gli indici per colonne dei valori 
                           %che soddisfano la condizione
        for i = 0:length(I)-1
            tX(:,I(length(I)-i)) = []; %Rimozione
        end
        %size(tX)

        %Calcolo LDA
        MdlLinear = fitcdiscr(tX',string(tY),'DiscrimType','pseudolinear');

        %Calcolo dell'importanza delle varie dimensioni della selezione
        %(calcolo delle componenti della base ridotta nella base originale)
        W = [];
        for i = 2:length(MdlLinear.Coeffs)
            for j = 1:i-1
                if isempty(W)
                    l = MdlLinear.Coeffs(i,j).Linear;
                    %Calcolo del modulo e normalizzazione
                    l = abs(l);
                    l = l/sum(l);
                    W = l;
                else
                    l = MdlLinear.Coeffs(i,j).Linear;
                    %Calcolo del modulo e normalizzazione
                    l = abs(l);
                    l = l/sum(l);
                    W = W+l; %(Come sopra, ma qui W non empty)
                end
            end
        end
        
        %Reinserimento delle dimensioni con varianza nulla
        for i = 1:length(I)
            W = cat(1,W(1:I(i)-1),0,W(I(i):length(W)));
        end
        
        %Calcolo del modulo e normalizzazione
        W = abs(W);
        W = W/sum(W);

    else %Si procede in ogni altro caso come se fosse sempre 
         %"if strcmp(INITIAL_IMPORTANCE,'LRP')"

        %Ricerca del predecision layer
        for i = 1:length(lgraph.Layers)
            if strcmp(lgraph.Layers(i).Name,PREDECISION_FC)
                layer = lgraph.Layers(i);
            end
        end

        W = sum(abs(layer.Weights),1)';
        W = W/sum(W);
    end

    %Plot della curva e della treshhold
    B = reshape(W,[],1);
    B = abs(B);
    B = sort(B,1);
    cB = cumtrapz(B); %Calcolo dell'integrale cumulativo approssimato con il metodo dei trapezi
    totB = cB(length(B));
    ix = find(cB>(totB*TRESHOLD),1);
    T = double(B(ix-1));
    length(cB);
    
    if PLOT_OPTIONAL %Aggiornamento della figura
        figure
        plot(B)
        yline(T,'-','Threshold');
    end

    %Propagazione delle importanze ai layer precedenti usando la regola 
    %dell'LRP: dividere l'importanza dei nodi di output fra i nodi di input
    %a seconda dei contributi (|attivazione*collegamento|) e salvataggio 
    %delle importanze su una mappa M.
    %Le importance salvate per ogni layer risultano una percentuale
    %rispetto all'importanza finale, quindi per ogni layer la somma delle 
    %importanze è 1.
    M = containers.Map;
    M('fc') = W;

    %Ricerca dell'indice del predecision layer
    for i = 1:length(lgraph.Layers)
        if strcmp(lgraph.Layers(i).Name,PREDECISION_FC)
            I = i;
        end
    end

    %Per ogni nodo si parte da quello di indice i = I-1 e si decrementa 
    %l'indice di 1 fino a raggiungere il layer di indice 1
    for i = I-1:-1:1
        
        layer = lgraph.Layers(i); %Layer di indice i 

        if isequal(class(layer),'nnet.cnn.layer.DropoutLayer') || ...
           isequal(class(layer),'nnet.cnn.layer.ReLULayer') ||...
           isequal(class(layer),'nnet.cnn.layer.MaxPooling2DLayer') ||...
           isequal(class(layer),'nnet.cnn.layer.CrossChannelNormalizationLayer')

            %Per questo tipo di layer le importanze rimangono invariate
            %(anche per il MaxPooling perché si sta calcolando l'importanza
            %dei canali e non dei singoli nodi, dovendo fare pruning sui canali)
            M(layer.Name) = W;

        elseif isequal(class(layer),'nnet.cnn.layer.FullyConnectedLayer')

            %Recupero del nome del layer precedente per ottenere le attivazioni
            %di quel layer (salvate come somma sulla mappa A)
            prev = lgraph.Connections(strcmp(lgraph.Connections.Destination,layer.Name),:);
            prev = string(prev.Source);

            X = reshape(A(prev),[],1); %Reshape utile in caso il fc seguisse un layer conv

            %Propagazione delle importanze
            Y = abs(layer.Weights)*X;
            W = (abs(layer.Weights)'*(W./Y)).*X;

            W = reshape(W,size(A(prev))); %Ripristiono della forma originale

            %Eliminazione della dimensione spaziale (per fare pruning sui canali)
            if length(size(W))>2
                W = sum(W,[1,2]);             
            end
            %size(W)

            M(layer.Name) = W;

        elseif isequal(class(layer),'nnet.cnn.layer.GroupedConvolution2DLayer') ||...
               isequal(class(layer),'nnet.cnn.layer.Convolution2DLayer')

            %Recupero del nome del layer precedente per ottenere le attivazioni
            %di quel layer (salvate come somma sulla mappa A)
            prev = lgraph.Connections(strcmp(lgraph.Connections.Destination,layer.Name),:);
            prev = string(prev.Source);

            X = A(prev);
            s = size(X);
            X = reshape(X,s(1),s(2),s(3),1);
            X = dlarray(X,'SSCB');
            %size(layer.Weights)

            %Propagazione delle importanze
            %Deep learning convolution: l'operazione di convoluzione 
            %applica filtri scorrevoli ai dati di input
            Y = dlconv(X,abs(layer.Weights),0,...
                    'Stride',layer.Stride,...
                    'DilationFactor',layer.DilationFactor,...
                    'Padding',layer.PaddingSize,...
                    'PaddingValue',layer.PaddingValue);
            s = size(Y); %Recupero delle dimensioni di Y
            W = repmat(W/s(1)/s(2),s(1),s(2)); %Ripetere (W/s(1)/s(2)) s(1) ed s(2) volte nelle due dimensioni
            W = reshape(W,s(1),s(2),s(3),1);
            W = dlarray(W,'SSCB');
            %Deep learning transposed convolution: calcola la convoluzione 
            %trasposta dati di input
            W = dltranspconv((W./Y),abs(layer.Weights),0,...
                    'Stride',layer.Stride,...
                    'DilationFactor',layer.DilationFactor,...
                    'Cropping',layer.PaddingSize);
            W = W.*X; %Equivale a W(i,j)=W(i,j)*X(i,j) con i,j rispettivamente generico elemento di riga e colonna
            W = extractdata(sum(W,[1,2]));
            %size(W)

            M(layer.Name) = W;

        elseif isequal(class(layer),'GroupConv')
            
            %Recupero del nome del layer precedente per ottenere le attivazioni
            %di quel layer (salvate come somma sulla mappa A)
            prev = lgraph.Connections(strcmp(lgraph.Connections.Destination,layer.Name),:);
            prev = string(prev.Source);
            X = A(prev);
            %s = size(X);
            %size(W)

            w1 = layer.Net1.Layers(1).Weights; %Get Net1 layers -> get Net1 weights
            s1 = size(w1); %Recupero delle dimensioni di w1
            w2 = layer.Net2.Layers(1).Weights; %Get Net2 layers -> get Net2 weights
            s2 = size(w2); %Recupero delle dimensioni di w2

            X1 = X(:,:,1:s1(3));
            X2 = X(:,:,s1(3)+1:s1(3)+s2(3));
            W1 = W(:,:,1:s1(4));
            W2 = W(:,:,s1(4)+1:s1(4)+s2(4));

            X1 = dlarray(X1,'SSCB');
            X2 = dlarray(X2,'SSCB');

            l1 = layer.Net1.Layers; %Get Net1 layers
            l2 = layer.Net2.Layers; %Get Ne21 layers

            %Propagazione delle importanze
            %Deep learning convolution: l'operazione di convoluzione 
            %applica filtri scorrevoli ai dati di input
            Y1 = dlconv(X1,abs(w1),0,...
                    'Stride',l1.Stride,...
                    'DilationFactor',l1.DilationFactor,...
                    'Padding',l1.PaddingSize,...
                    'PaddingValue',l1.PaddingValue);
            s = size(Y1); %Recupero delle dimensioni di Y1
            W1 = repmat(W1/s(1)/s(2),s(1),s(2)); %Ripetere (W1/s(1)/s(2)) s(1) ed s(2) volte nelle due dimensioni
            W1 = reshape(W1,s(1),s(2),s(3),1);
            W1 = dlarray(W1,'SSCB');
            %Deep learning transposed convolution: calcola la convoluzione 
            %trasposta dati di input
            W1 = dltranspconv((W1./Y1),abs(w1),0,...
                    'Stride',l1.Stride,...
                    'DilationFactor',l1.DilationFactor,...
                    'Cropping',l1.PaddingSize);
            W1 = W1.*X1; %Equivale a W1(i,j)=W1(i,j)*X1(i,j) con i,j rispettivamente generico elemento di riga e colonna
            W1 = extractdata(sum(W1,[1,2]));

            %Analogo a quanto appena svolto
            Y2 = dlconv(X2,abs(w2),0,...
                    'Stride',l2.Stride,...
                    'DilationFactor',l2.DilationFactor,...
                    'Padding',l2.PaddingSize,...
                    'PaddingValue',l2.PaddingValue);
            s = size(Y2);
            W2 = repmat(W2/s(1)/s(2),s(1),s(2));
            W2 = reshape(W2,s(1),s(2),s(3),1);
            W2 = dlarray(W2,'SSCB');
            W2 = dltranspconv((W2./Y2),abs(w2),0,...
                    'Stride',l2.Stride,...
                    'DilationFactor',l2.DilationFactor,...
                    'Cropping',l2.PaddingSize);
            W2 = W2.*X2;
            W2 = extractdata(sum(W2,[1,2]));

            W = cat(3,W1,W2); %Concatenazione

            M(layer.Name) = W;
        end
    end

    %Pruning della rete in base alla treshold ed alle importanze calcolate 
    %sui layer specificati da LAYERS_TO_PRUNE
    
    %l=lgraph.Layers;
    %Ripetere per tutti i layer da potare
    for nl = 1:length(LAYERS_TO_PRUNE)

        try
            pName = LAYERS_TO_PRUNE{nl};
        catch ME
            pName = LAYERS_TO_PRUNE(nl);
        end


        if ~isstring(pName)&&~ischar(pName) %Se pName non è una sequenza di caratteri
            pName = pName.Name; %allora e' stato passato direttamente il layer
        end

        %Ricerca del layer da potare
        for i = 1:length(lgraph.Layers)
            if strcmp(lgraph.Layers(i).Name, pName)
                I = i;
                layer = lgraph.Layers(i);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Pruning del layer: rimozione degli input non necessari

        if isequal(class(layer),'nnet.cnn.layer.FullyConnectedLayer')

            W = M(layer.Name);
            %size(W)

            %Calcolo della treshold
            B = reshape(W,[],1);
            B = abs(B);
            B = sort(B);
            cB = cumtrapz(B); %Calcolo dell'integrale cumulativo approssimato con il metodo dei trapezi
            totB = cB(length(B));
            ix = find(cB>(totB*TRESHOLD),1);
            T = double(B(ix-1));
            length(cB);
            if PLOT_OPTIONAL %Aggiornamento della figura
                figure
                plot(B)
                yline(T,'-','Threshold');
            end

            %Ricerca dei valori sotto alla treshold
            II = find(W<T);

            %Rimozione delle colonne equivalenti nella matrice dei pesi
            s1 = size(W);
            w = layer.Weights;
            s2 = size(w);

            w = reshape(w,s2(1),[],s1(length(s1)));
            s = size(w);

            if length(s)>2
                w(:,:,II) = [];
            else 
                w(:,II) = [];
            end

            %Creazione di un nuovo layer con i pesi aggiornati e 
            %sostituzione in lgraph
            s = size(w);
            w = reshape(w,s(1),[]);
            newLayer = fullyConnectedLayer(s(1),...
                    'Name',layer.Name,...
                    'Weights',w,...
                    'Bias',layer.Bias); 


        elseif isequal(class(layer),'nnet.cnn.layer.Convolution2DLayer')
            
            W = M(layer.Name);
            %size(W)
            w = layer.Weights;

            %Calcolo della treshold
            B = reshape(W,[],1);
            B = abs(B);
            B = sort(B);
            cB = cumtrapz(B); %Calcolo dell'integrale cumulativo approssimato con il metodo dei trapezi
            totB = cB(length(B));
            ix = find(cB>(totB*TRESHOLD),1);
            T = double(B(ix-1));
            length(cB);
            if PLOT_OPTIONAL %Aggiornamento della figura
                figure
                plot(B)
                yline(T,'-','Threshold');
            end

            %Ricerca dei valori sotto alla treshold
            II = find(W<T);

            %Rimozione delle colonne equivalenti nella matrice dei pesi
            w(:,:,II,:) = [];

            s = size(w);
            
            %Creazione di un nuovo layer con i pesi aggiornati e 
            %sostituzione in lgraph
            newLayer = convolution2dLayer(layer.FilterSize,s(4),...
                'Name',layer.Name,...
                'Weights',w,...
                'Bias',layer.Bias,...
                'Stride',layer.Stride,...
                'DilationFactor',layer.DilationFactor,...
                'Padding',layer.PaddingSize,...
                'PaddingValue',layer.PaddingValue);


        elseif isequal(class(layer),'nnet.cnn.layer.GroupedConvolution2DLayer')
            
            W = M(layer.Name);
            %size(W)
            s = size(layer.Weights);

            %Calcolo della treshold
            B = reshape(W,[],1);
            B = abs(B);
            B = sort(B);
            cB = cumtrapz(B); %Calcolo dell'integrale cumulativo approssimato con il metodo dei trapezi
            totB = cB(length(B));
            ix = find(cB>(totB*TRESHOLD),1);
            T = double(B(ix-1));
            length(cB);
            if PLOT_OPTIONAL %Aggiornamento della figura
                figure
                plot(B)
                yline(T,'-','Threshold');
            end

            %Ricerca dei valori sotto alla treshold
            %Rimozione delle colonne equivalenti nella matrice dei pesi
            W = reshape(W,[],s(5));
            w = {};
            b = {};
            for j = 1:s(5)
                II = find(W(:,j)<T);

                wj = layer.Weights(:,:,:,:,j);
                bj = layer.Bias(:,:,:,j);

                wj(:,:,II,:) = [];

                w(j) = {wj};
                b(j) = {bj};
            end
            W = reshape(W,[],1);

            %Creazione di un nuovo layer con i pesi aggiornati e 
            %sostituzione in lgraph
            newLayer = GroupConv(layer,w,b);


        elseif isequal(class(layer),'GroupConv')
            W = M(layer.Name);
            w1 = layer.Net1.Layers(1).Weights;
            w2 = layer.Net2.Layers(1).Weights;
            b1 = layer.Net1.Layers(1).Bias;
            b2 = layer.Net2.Layers(1).Bias;
            s1 = size(w1);
            s2 = size(w2);
            W = reshape(W,[],1);
            W1 = W(1:s1(3));
            W2 = W(s1(3)+1:s1(3)+s2(3));

            %Calcolo della treshold
            B = reshape(W,[],1);
            B = abs(B);
            B = sort(B);
            cB = cumtrapz(B); %Calcolo dell'integrale cumulativo approssimato con il metodo dei trapezi
            totB = cB(length(B));
            ix = find(cB>(totB*TRESHOLD),1);
            T = double(B(ix-1));
            length(cB);
            if PLOT_OPTIONAL %Aggiornamento della figura
                figure
                plot(B)
                yline(T,'-','Threshold');
            end

            %Ricerca dei valori sotto alla treshold
            II1 = find(W1<T);
            II2 = find(W2<T);

            %Rimozione delle colonne equivalenti nelle matrici dei pesi
            w1(:,:,II1,:) = [];
            w2(:,:,II2,:) = [];

            w = {};
            b = {};
            w(1) = {w1};
            b(1) = {b1};
            w(2) = {w2};
            b(2) = {b2};

            %Creazione di un nuovo layer con i pesi aggiornati e 
            %sostituzione in lgraph
            newLayer = GroupConv(layer.OriginalLayer,w,b);

        else
            continue
        end

        %Sostituzione in lgraph
        lgraph = replaceLayer(lgraph,layer.Name,newLayer);


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Rimozione degli output non necessari dal layer precedente

        %Per ogni nodo si parte da quello di indice i = I-1 e si decrementa 
        %l'indice di 1 fino a raggiungere il layer di indice 1
        %(Operazioni analoghe a quelle descritte sopra:
        %   Utilizzo della treshold precedentemente calcolata
        %   Ricerca dei valori sotto alla treshold
        %   Rimozione delle colonne equivalenti nella matrice dei pesi
        %   Creazione di un nuovo layer con i pesi aggiornati e sostituzione in lgraph)
        for i = I-1:-1:1
            layer = lgraph.Layers(i);
            if isequal(class(layer),'nnet.cnn.layer.FullyConnectedLayer')
                II = find(W<T);
                w = layer.Weights;
                b = layer.Bias;
                w(II,:) = [];
                b(II,:) = [];

                s = size(w);
                newLayer = fullyConnectedLayer(s(1),...
                    'Name',layer.Name,...
                    'Weights',w,...
                    'Bias',b);

                break
            elseif isequal(class(layer),'nnet.cnn.layer.Convolution2DLayer')
                %size(W)
                s = size(layer.Weights);
                w = layer.Weights;
                b = layer.Bias;
                II = find(W<T);

                w(:,:,:,II) = [];
                b(:,:,II) = [];

                s = size(w);
                newLayer = convolution2dLayer(layer.FilterSize,s(4),...
                    'Name',layer.Name,...
                    'Weights',w,...
                    'Bias',b,...
                    'Stride',layer.Stride,...
                    'DilationFactor',layer.DilationFactor,...
                    'Padding',layer.PaddingSize,...
                    'PaddingValue',layer.PaddingValue);

                break
            elseif isequal(class(layer),'nnet.cnn.layer.GroupedConvolution2DLayer')
                %size(W)
                s = size(layer.Weights);

                W = reshape(W,[],s(5));
                w = {};
                b = {};
                for j = 1:s(5)
                    II = find(W(:,j)<T);

                    wj = layer.Weights(:,:,:,:,j);
                    bj = layer.Bias(:,:,:,j);

                    wj(:,:,:,II) = [];
                    bj(:,:,II) = [];

                    w(j) = {wj};
                    b(j) = {bj};
                end

                newLayer = GroupConv(layer,w,b);

                break
            elseif isequal(class(layer),'GroupConv')
                %size(W)
                w1 = layer.Net1.Layers(1).Weights;
                w2 = layer.Net2.Layers(1).Weights;
                b1 = layer.Net1.Layers(1).Bias;
                b2 = layer.Net2.Layers(1).Bias;
                s1 = size(w1);
                s2 = size(w2);
                W = reshape(W,[],1);
                W1 = W(1:s1(4));
                W2 = W(s1(4)+1:s1(4)+s2(4));

                II1 = find(W1<T);
                II2 = find(W2<T);

                w1(:,:,:,II1) = [];
                b1(:,:,II1) = [];

                w2(:,:,:,II2) = [];
                b2(:,:,II2) = [];


                w = {};
                b = {};
                w(1) = {w1};
                b(1) = {b1};
                w(2) = {w2};
                b(2) = {b2};

                newLayer = GroupConv(layer.OriginalLayer,w,b);

                break
            end

        end
        
        %Sostituzione in lgraph
        lgraph = replaceLayer(lgraph,layer.Name,newLayer);
        
    end

    dlnet = dlnetwork(lgraph); %Conversione lgraph -> dlnetwork
    
    
    %Test delle reti:
    %   Rete originale/di partenza
    Z = predict(net,testImages); %Operazione di predict
    loss = extractdata(crossentropy(Z,yy)) %Calcolo loss
    E = sum(abs(Z-yy)/2,1);
    accuracy = extractdata(1-sum(E)/length(E)) %Calcolo accuracy
    
    %   Rete potata
    Z = predict(dlnet,testImages); %Operazione di predict
    loss_pruned = extractdata(crossentropy(Z,yy)) %Calcolo loss
    E = sum(abs(Z-yy)/2,1);
    accuracy_pruned = extractdata(1-sum(E)/length(E)) %Calcolo accuracy

    drawnow

end