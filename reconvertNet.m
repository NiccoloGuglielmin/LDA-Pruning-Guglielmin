%{

clc %Elimina tutto il testo dalla Command Window
clear all %Elimina tutte le variabili contenute nel Workspace
close all %Chiude tutte le Figure aperte

%Caricamento della rete originale/di partenza
load('NetAndData.mat','netTransfer');

%Caricamento rete modificata
load('ModifiedNet');

%}

%Conversione della rete da oggetto di tipo dlnetwork in un oggetto 
%LayerGraph, incluso il classificationLayer di output
function [lgraph] = reconvertNet(dlnet)

    %Caricamento della rete originale/di partenza
    load('NetAndData.mat','netTransfer');

    %analyzeNetwork(dlnet)

    %Crea un oggetto contenente la rete di partenza convertita in tipo layerGraph
    %lgraph = layerGraph(netTransfer.Layers(1:length(netTransfer.Layers)-1));
    %Rispetto al commento manca il -1 in quanto si deve aggiungere il layer di
    %output (classificationLayer)
    lgraph = layerGraph(netTransfer.Layers(1:length(netTransfer.Layers)));

    %Per ogni layer nella rete
    for i=1:length(lgraph)

        %Get dei layer della rete di partenza
        layer = lgraph.Layers(i);
        mlayers = dlnet.Layers; %Get dei layer della rete da convertire

        %Per ogni layer della rete modificata
        for j = 1:length(mlayers)
            %Per ottenere il layer modificato dalla rete da convertire
            if strcmp(mlayers(j).Name,layer.Name)
                modLayer = mlayers(j);
                break
            end
        end

        %Creazione di un nuovo layer che, rispetto al layer corrente della rete
        %di partenza, ha weights e bias impostati come quelli del
        %corrispondente layer modificato nella rete da convertire
        %(notare che si copiano tutte le informazioni del layer layer, tranne 
        %proprio Weights e Bias che si copiano da modLayer)
        if isequal(class(layer),'nnet.cnn.layer.Convolution2DLayer')
            newLayer = convolution2dLayer(layer.FilterSize,layer.NumFilters,...
                    'Name',layer.Name,...
                    'Weights',modLayer.Weights, ...
                    'Bias',modLayer.Bias,...
                    'Stride',layer.Stride,...
                    'DilationFactor',layer.DilationFactor,...
                    'Padding',layer.PaddingSize,...
                    'PaddingValue',layer.PaddingValue);

        elseif isequal(class(layer),'nnet.cnn.layer.GroupedConvolution2DLayer')
            newLayer = groupedConvolution2dLayer(layer.FilterSize,layer.NumFiltersPerGroup,layer.NumGroups,...
                    'Name',layer.Name,...
                    'Weights',modLayer.Weights,...
                    'Bias',modLayer.Bias,...
                    'Stride',layer.Stride,...
                    'DilationFactor',layer.DilationFactor,...
                    'Padding',layer.PaddingSize,...
                    'PaddingValue',layer.PaddingValue);

        elseif isequal(class(layer),'nnet.cnn.layer.FullyConnectedLayer')
            newLayer = fullyConnectedLayer(layer.OutputSize,...
                    'Name',layer.Name,...
                    'Weights',modLayer.Weights,...
                    'Bias',modLayer.Bias);

        elseif isequal(class(layer),'GroupConv')
            w1 = modLayer.Net1.Layers(1).Weights;
            w2 = modLayer.Net2.Layers(1).Weights;
            b1 = modLayer.Net1.Layers(1).Bias;
            b2 = modLayer.Net2.Layers(1).Bias;

            w = {};
            b = {};
            w(1) = {w1};
            b(1) = {b1};
            w(2) = {w2};
            b(2) = {b2};

            newLayer = GroupConv(layer.OriginalLayer,w,b);  
                
        else
            %Nel caso il layer corrente layer non sia un tipo di layer modificato
            newLayer = layer;

        end

        %Si rimpiazza il layer della rete di partenza con quello nuovo modificato
        lgraph = replaceLayer(lgraph,layer.Name,newLayer);

    end

    %figure
    %plot(lgraph)

    %dlnet = dlnetwork(lgraph)

    %Salvataggio della rete convertita in lgraph (con aggiunta del
    %classificationLayer di output)
    save('ReconvertedNet','lgraph')

end
