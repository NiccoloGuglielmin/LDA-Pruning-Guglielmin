%Preparazione variabili tX, tY, A per pruning LDA effettivo
%Parametri: rete da fare pruning, numero di epochs di dati (per prendere le
%           attivazioni) >=10, nome del layer fc pre-softmax
function [tX,tY,A] = pruneA(dlnet,EPOCHS,PREDECISION_INPUT)

    %Caricamento rete e dataset
    load('NetAndData.mat');
    miniBatchSize = 64;
    mbq = minibatchqueue(trainingImages,'MiniBatchSize',miniBatchSize);

    %Ricerca dell'indice del layer (ultima attivazione) di interesse
    layers = dlnet.Layers;
    for i = 1:length(layers)
        if strcmp(layers(i).Name,PREDECISION_INPUT)
            I = i;
        end    
    end
    
    %Creazione di matrice vuota di dimensione 1xI
    list = cell(1,I);
    for i = 1:I
        %Salvataggio dei nomi dei layer nella matrice list
        list{i} = dlnet.Layers(i).Name;
    end
    
    %Inizializzazione variabili
    tX = [];
    tY = [];
    A = containers.Map; %Oggetto mappa: le chiavi keys (non necessariamente interi) indicizzano valori values
    
    %Per ogni epoca di pruning
    for n = 1:EPOCHS
        
        disp(strcat("    Computing epoch ",int2str(n)));
        shuffle(mbq) %Ritorna un oggetto datastore contenente un ordine casuale dei dati contenuti in mbq
        while hasdata(mbq) %Per ogni dato contenuto in mbq
            
            %Ottiene il prossimo mini-batch dei dati contenuti in mbq
            %next(mbq) ritorna tante variabili quante il numero di output di mbq
            [dlX,dlY] = next(mbq);
            %Etichetta i dati contenuti in dlX in base al formato dei dati SSCB
            %S —> Spatial / C —> Channel / B —> Batch observations / 
            %T —> Time or sequence / U —> Unspecified
            dlX = dlarray(dlX,'SSCB');

            %Inizializzazione variabile X
            X = {};
            [X{1:I}] = predict(dlnet,dlX,'Outputs',list); %Operazione di predict
            
            if isempty(tY) %Alla prima iterazione (n == 1)
                tY = dlY;
                tX = gather(extractdata(X{I-1})); %Calcolo dei dati in input e salvataggio in tX
                A('input') = sum(abs(gather(extractdata(dlX))),length(size(dlX)));
            else %2 <= n <= EPOCHS
                %Concatena dlY all'estremità di tY lungo la dimensione 1 quando tY e dlY hanno dimensioni compatibili 
                %(le lunghezze delle dimensioni corrispondono tranne che per la dimensione operativa 1).
                tY = cat(1,tY,dlY);
                %Analogamente a prima: operazione di concatenazione di array
                tX = cat(2,tX,gather(extractdata(X{I-1})));
                A('input') = A('input')+sum(abs(gather(extractdata(dlX))),length(size(dlX)));
            end
            
            %Aggiornamento contenuto di A
            for i = 1:I
                %strcmp ritorna 1 (true) oppure 0 (false)
                %any determina se qualche valore restituito da strcmp è nonzero
                %~any(strcmp(keys(A),list{i})) determina se nessun valore restituito da strcmp è nonzero
                if ~any(strcmp(keys(A),list{i}))
                    A(list{i}) = sum(abs(gather(extractdata(X{i}))),length(size(X{i})));
                else %Altrimenti se qualche valore restituito da strcmp è nonzero
                    A(list{i}) = A(list{i})+sum(abs(gather(extractdata(X{i}))),length(size(X{i})));
                end
            end
        end
    end
    disp("");

    %Raccoglie tutti gli elementi di tY
    tY = gather(extractdata(tY));
    
    %disp(tX);
    %disp(tY);
    %disp(A);

end