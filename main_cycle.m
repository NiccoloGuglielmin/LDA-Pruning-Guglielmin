function main_cycle(PRUNING_PLOT,...
                    PRUNING_ITERATIONS,...
                    TRAINING_EPOCHS,...
                    BATCH_SIZE,...
                    PRUNING_DATA_EPOCHS,...
                    PERCENTAGE_TO_PRUNE,...
                    INITIAL_IMPORTANCE,...
                    PLOT_OPTIONAL,...
                    plotLine,...
                    tacc)

    %Caricamento rete originale/di partenza
    OriginalNet = 'NetAndData.mat';
    N = loadNet(OriginalNet); %Output di tipo dlnetwork
    [ONParams,Oaccuracy,Oloss] = computeScore(N); %Valutazione delle prestazioni
    
    %Per salvare la rete originale/di partenza alla fine del ciclo e 
    %poterla confrontare con la rete finale potata
    OriginalNet = reconvertNet(N);
    
    %for i = 1:PRUNING_ITERATIONS
    i = 1;
    accuracy = Oaccuracy;
    while(i >= 0 & i <= PRUNING_ITERATIONS & tacc > 0 & accuracy >= tacc)

        analyzeNetwork(N);%

        %%%%%%%%%%%%%%%%%%%%%%%%%% STEP 1 -> PRUNING %%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [NParams,accuracy,loss] = computeScore(N); %Valutazione delle prestazioni

        disp(strcat("Computing iteration ",int2str(i)));
        
        %Aggiornamento del grafico di pruning
        figure(PRUNING_PLOT)
        hold on
        addpoints(plotLine,NParams/ONParams,double((accuracy-Oaccuracy)*100/Oaccuracy))
        %plot(NParams/ONParams,(accuracy-Oaccuracy)*100/Oaccuracy,dot);
        drawnow

        %Preparazione variabili per pruning LDA effettivo
        [tX,tY,A] = pruneA(N,PRUNING_DATA_EPOCHS,'fc');

        %(Iniziare l'esecuzione del pruning dal terzo layer, non dall'input del primo conv layer)
        layersToPrune = N.Layers(3:length(N.Layers));

        %Pruning LDA
        [N,L,L0]= pruneB(N,tX,tY,A,PERCENTAGE_TO_PRUNE,'fc',INITIAL_IMPORTANCE,layersToPrune,PLOT_OPTIONAL);

        %%%%%%%%%%%%%%%%%%%%%%%% STEP 2 -> RE-TRAINING %%%%%%%%%%%%%%%%%%%%%%%%

        %{
        disp("Start net conversion");
        %Conversione della rete (con rimozione del layer di output classificationLayer)
        N = modifyNet(N); % ==> conversione in dlnetwork (NON SERVE!!! N già di tipo dlnetwork!)
        disp("End net conversion");
        %}
        %{
        disp("    Start net retraining");
        %Retraining della rete
        N = retraining(N,TRAINING_EPOCHS,i); %N dev'essere di tipo dlnetwork: dlnetwork ==> dlnetwork
        disp("    End net retraining");
        %}
        %{
        disp("Start net reconversion");
        %Riconversione della rete (con aggiunta del layer di output classificationLayer)
        N = reconvertNet(N); % ==> conversione in dlnetwork (NON SERVE!!! N
        %deve rimanere di tipo dlnetwork per il prossimo calcolo di score!)
        N = dlnetwork(N);
        disp("End net reconversion");
        %}

        i = i + 1;
    end
    
    analyzeNetwork(N);%
    
    [NParams,accuracy,loss] = computeScore(N); %Valutazione delle prestazioni

    %Aggiornamento del grafico di pruning
    figure(PRUNING_PLOT)
    hold on
    addpoints(plotLine,NParams/ONParams,double((accuracy-Oaccuracy)*100/Oaccuracy))
    %plot(NParams/ONParams,(accuracy-Oaccuracy)*100/Oaccuracy,dot);
    drawnow
    
    %Per salvare la rete potata per poterla confrontare con la rete originale/di partenza
    PrunedNet = reconvertNet(N);
    
    analyzeNetwork(OriginalNet);%
    analyzeNetwork(PrunedNet);%
    
    %Salvataggio della rete originale/di partenza e quella potata (di tipo LayerGraph)
    save('OriginalAndPrunedNets','OriginalNet','PrunedNet');
    
end

%Funzione che carica la rete e la converte in un oggetto dlnetwork che
%rappresenta una rete neurale profonda modificabile
function dlnet = loadNet(NomeFile) %Output di tipo dlnetwork

    if strcmp(NomeFile,'NetAndData.mat')
        load('NetAndData.mat','netTransfer');
        n = netTransfer;
        %Se nella rete è presente il layer di output
        if isequal(class(n.Layers(length(n.Layers))),'nnet.cnn.layer.ClassificationOutputLayer')
            %Non lo considero
            g = layerGraph(n.Layers(1:length(n.Layers)-1));
        else %Altrimenti converto tutta la rete
            g = layerGraph(n.Layers(1:length(n.Layers)));
        end
        dlnet = dlnetwork(g);
    else
        dlnet = load(NomeFile,'dlnet');
        dlnet = dlnet.('dlnet');
    end
    
end

%Funzione che calcola lo score adattando il tipo di rete che verrà passato
%in input alla funzione score in score.m
function [NParamsNet,accuracyNet,lossNet] = computeScore(net)

    %Se la rete passata in input non è di tipo dlnetwork allora convertire
    %(dlnetwork accetta in input solo oggetti di tipo LayerGraph oppure
    %Layer array; net deve avere il layer di input)
    if isa(net,'SeriesNetwork')
        %Se nella rete è presente il layer di output
        if isequal(class(n.Layers(length(n.Layers))),'nnet.cnn.layer.ClassificationOutputLayer')
            %Non lo considero
            net = layerGraph(net.Layers(1:length(net.Layers)-1));
        else %Altrimenti converto tutta la rete
            net = layerGraph(net.Layers(1:length(net.Layers)));
        end
    elseif ~isa(net,'dlnetwork') %Caso generale (in caso di errore ==> eccezione)
        net = dlnetwork(net)
    end %Se isa(net,'dlnetwork') non serve eseguire alcun comando: calcola score direttamente
    
    %Calcolo score della rete
    %Deve ricevere in input una dlnetwork
    [NParamsNet,accuracyNet,lossNet] = score(net)

end