%{
                    UNIVERSITA' DEGLI STUDI DI PADOVA
               DIPARTIMENTO DI INGEGNERIA DELL'INFORMAZIONE
                CORSO DI LAUREA IN INGEGNERIA INFORMATICA
                    
                    POTATURA DI RETI NEURALI PROFONDE 
                  MEDIANTE ANALISI DISCRIMINANTE LINEARE
    
    Relatore: Prof. Nanni Loris
    Laureando: Guglielmin Niccolò
    Correlatore: Dott. Maguolo Gianluca

                        ANNO ACCADEMICO 2020-2021

    Codice sviluppato con MATLAB R2021a - academic use
    Versione: 9.10.0.1710957 (R2021a) Update 4
%}

%warning off %Disattiva i warning
clc %Elimina tutto il testo dalla Command Window
clear all %Elimina tutte le variabili contenute nel Workspace
close all %Chiude tutte le Figure aperte

%{
    Algorithm 1: Deep LDA Pruning of Neural Network

    Input: base net, acceptable accuracy 𝑡𝑎𝑐𝑐
    Result: task-desirable pruned models
    Pre-train: SGD optimization with cross entropy loss, L2-regularization and Dropout.

    while accuracy ⩾ 𝑡𝑎𝑐𝑐 do
        Step 1 → Pruning
            1. Task Utility Unravelling from Final Latent Space
            2. Cross-Layer Task Utility Tracing via Deconv
            3. Pruning as Utility Thresholding
        Step 2 → Re-training
            Similar to the pre-training step. Save model if needed.
    end
%}

%Salvataggio dell'output della command window indirizzato in un file pdf
diary on
diary('PDF_Plot/cw_output.txt')

%Inizio pruning
disp("Start pruning");

%Impostazione del grafico di pruning
PRUNING_PLOT = figure('Name','pruningPlot');
title("Pruning")
xlabel("% Parameters")
ylabel("Accuracy")
set(gca,'XDir','reverse')
xlim([-0.1 1.1])
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%iperparametri
PRUNING_ITERATIONS = 7;%2;%5;%15;
TRAINING_EPOCHS = 10;%2;%10;
BATCH_SIZE = 128; %I risultati migliori si ottengono con 64 o 128 in qaunto più stabile
PRUNING_DATA_EPOCHS = 10;%2;%10;
PERCENTAGE_TO_PRUNE = 0.01; %1 elimina l'intera rete
INITIAL_IMPORTANCE = 'LDA';%'LRP'; %LDA o LRP
PLOT_OPTIONAL = false;
tacc = 0.5000; %Accuracy sotto alla quale il pruning termina. Formato: 0 <= tacc <= 1. 1 rappresenta 100%

lineLossTrain = animatedline('Color',[1 0 0]); %Crea curva a tratti
%--------------------------------------------------------
try
    main_cycle(PRUNING_PLOT,...
            PRUNING_ITERATIONS,...
            TRAINING_EPOCHS,...
            BATCH_SIZE,...
            PRUNING_DATA_EPOCHS,...
            PERCENTAGE_TO_PRUNE,...
            INITIAL_IMPORTANCE,...
            PLOT_OPTIONAL,...
            lineLossTrain,...
            tacc)
    %{
    main_cycle(PRUNING_PLOT,...
            CF,...
            PRUNING_ITERATIONS,...
            TRAINING_EPOCHS,...
            BATCH_SIZE,...
            SECOND_LOSS_MULTIPLIER,...
            PRUNING_DATA_EPOCHS,...
            PERCENTAGE_TO_PRUNE,...
            LEARNING_RATE,...
            INITIAL_IMPORTANCE,...
            PLOT_OPTIONAL,...
            CLEAR_GPU_MEMORY,...
            lineLossTrain)
    %}

catch exception %Cattura eccezione generica
   exception
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Salvataggio del grafico di pruning ottenuto
%saveas(PRUNING_PLOT,'PDF_Plot/pruningPlot.png');
saveas(PRUNING_PLOT,'PDF_Plot/pruningPlot.pdf');

%Fine pruning
disp("End pruning");

%Fine salvataggio dell'output della command window
diary off

%{
Todelete
    %carica dati
    load('AlexNetTrainedWorkspace.mat');

    net_to_prune = netTransfer;  %load net
    %All variables in workspace?

    tacc = -1; %Initialize tacc (delete -1)

    %Algorithm 1: Deep LDA Pruning of Neural Network
    %Input: base net, acceptable accuracy 𝑡𝑎𝑐𝑐 or model complexity 𝑡𝑐𝑜𝑚
    layersTransfer = net_to_prune.Layers(1:end);
    layers = [
        layersTransfer
             ]
    lgraph = layerGraph;
    lgraph = addLayers(lgraph,layers); %Base net pre-trained
    disp(lgraph); %Output: lgraph's properties
    plot(lgraph); %Plot lgraph
    disp("here");
    disp(net_to_prune.Layers(10));
    disp(net_to_prune.Layers(10).Bias);
    disp(net_to_prune.Layers(10).Weights);
    disp("end");


    epsilon = 1;
    current_tacc = tacc;
    %current_tacc ∈ [tacc - epsilon, tacc + epsilon]
    %Pre-train: SGD optimization with cross entropy loss, L2-regularization, and Dropout. (OK)
    while ((current_tacc >= (tacc - epsilon)) & (current_tacc <= (tacc + epsilon))) %current_tacc >= tacc
        break
        %Step 1 → Pruning
            %1. Task Utility Unravelling from Final Latent Space (Section 3.1)
            %2. Cross-Layer Task Utility Tracing via Deconv (Section 3.2)
            %3. Pruning as Utility Thresholding (Section 3.3)
        %Step 2 → Re-training
            %Similar to the pre-training step. Save model if needed.
    end
    %Result: task-desirable pruned models (OK?)

%}