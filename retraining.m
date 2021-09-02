%{

clc %Elimina tutto il testo dalla Command Window
clear all %Elimina tutte le variabili contenute nel Workspace
close all %Chiude tutte le Figure aperte

%Caricamento della rete originale/di partenza
load('NetAndData.mat');

%Caricamento rete modificata
load('ModifiedNet');

%}

%Per implementare un training loop è necessario:
% - aver convertito la rete in un oggetto dlnetwork (senza aver includeso 
%   il layer di output
% - specificare la loss function
function [dlnet] = retraining(dlnet,epochs,numPruningIteration)

    %Caricamento del dataset
    load('NetAndData.mat');

    %Opzioni per il train del modello
    numEpochs = epochs;
    miniBatchSize = 10;

    numClasses = 10;
    mbq = minibatchqueue(trainingImages,'MiniBatchSize',miniBatchSize);

    %Opzioni per l'ottimizzazione SGDM
    initialLearnRate = 0.01;
    decay = 0.01;
    momentum = 0.9;

    %Impostazione del grafico di training
    TRAINING_PLOT = figure('Name','trainingPlot');
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on

    %Inizializzazione della variabile velocity per il SGDM solver
    velocity = [];

    iteration = 0;
    start = tic;

    %Training della rete. Per ogni epoch viene eseguito uno shuffle dei dati 
    %e, in seguito, un loop per ogni mini-batch dei dati. Per ogni mini-batch:
    % - valutare i gradienti, lo state e la loss del modello utilizzando le
    %   funzioni dlfeval e modelGradient ed aggiornare lo state della rete
    % - determinare il learning rate per il time-based decay learning rate schedule
    % - aggiornare i parametri della rete utilizzando la funzione sgdmupdate
    % - visualizzare il processo di training

    %Loop per tutte le epochs.
    for epoch = 1:numEpochs

        %Ritorna un oggetto datastore contenente un ordine casuale dei dati contenuti in mbq
        shuffle(mbq);

        %Loop sui vari mini-batch
        while hasdata(mbq)

            iteration = iteration+1;

            %Ottiene il prossimo mini-batch dei dati contenuti in mbq
            %next(mbq) ritorna tante variabili quante il numero di output di mbq
            [dlX, dlY] = next(mbq);
            dlY = (dlY == 1:numClasses)'.*1;
            dlX = dlarray(dlX,'SSCB');

            %Valutazione dei gradienti, stato e loss della rete utilizzando dlfeval 
            %e la funzione modelGradient, oltre all'aggiornamento dello stato della rete
            [gradients,state,loss] = dlfeval(@modelGradient,dlnet,dlX,dlY);
            dlnet.State = state;

            %Determinazione del learning rate per il time-based decay learning rate schedule
            learnRate = initialLearnRate/(1 + decay*iteration);

            %Aggiornamento dei  parametri della rete utilizzando ottimizzazione SGDM
            [dlnet,velocity] = sgdmupdate(dlnet,gradients,velocity,learnRate,momentum);

            %Visualizzazione del processo di retraining
            figure(TRAINING_PLOT)
            hold on
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,loss)
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow

        end

    end

    %Salvataggio del grafico di training ottenuto
    %saveas(TRAINING_PLOT,strcat('PDF_Plot/',num2str(numPruningIteration),'_trainingPlot.png'));
    saveas(TRAINING_PLOT,strcat('PDF_Plot/',num2str(numPruningIteration),'_trainingPlot.pdf'));
    
end