%Calcolo informazioni relative alla rete passata in input
%Parametri: rete di tipo deep learning network dlnetwork
function [NParams,accuracy,loss] = score(dlnet)

    %Caricamento rete e dataset
    load('NetAndData.mat');
    numClasses = 10;
    testImages = dlarray(cast(testImages,'single'),'SSCB');
    yy = (yy' == 1:numClasses)'.*1;

    %Test della rete dlnet
    Z = predict(dlnet,testImages); %Operazione di predict
    loss = extractdata(crossentropy(Z,yy)); %Calcolo loss
    E = sum(abs(Z-yy)/2,1);
    accuracy = extractdata(1-sum(E)/length(E)); %Calcolo accuracy

    %Variabile relativa al numero di parametri della rete
    NParams = 0;

    %Per ogni layer della rete
    for i = 1:size(dlnet.Layers)
        layer = dlnet.Layers(i); %Get layer
        class(layer); %Classificazione del layer

        %Calcolo dei parametri del layer i
        if isequal(class(layer),'nnet.cnn.layer.Convolution2DLayer')
            W = size(layer.Weights); %Get weights
            B = size(layer.Bias); %Get bias
            %Calcolo ed aggiunta dei parametri del layer corrente a quelli
            %calcolati in precedenza
            NParams = NParams+prod(W)+prod(B);
        elseif isequal(class(layer),'nnet.cnn.layer.GroupedConvolution2DLayer')
            W = size(layer.Weights); %Get weights
            B = size(layer.Bias); %Get bias
            %Calcolo ed aggiunta dei parametri del layer corrente a quelli
            %calcolati in precedenza
            NParams = NParams+prod(W)+prod(B);
        elseif isequal(class(layer),'nnet.cnn.layer.FullyConnectedLayer')
            W = size(layer.Weights); %Get weights
            B = size(layer.Bias); %Get bias
            %Calcolo ed aggiunta dei parametri del layer corrente a quelli
            %calcolati in precedenza
            NParams = NParams+prod(W)+prod(B);
        elseif isequal(class(layer),'GroupConv')
            L1 = layer.Net1.Layers; %Get Net1 layers
            L2 = layer.Net2.Layers; %Get Net2 layers
            W1 = size(L1.Weights); %Get weights
            B1 = size(L1.Bias); %Get bias
            W2 = size(L2.Weights); %Get weights
            B2 = size(L2.Bias); %Get bias
            %Calcolo ed aggiunta dei parametri del layer corrente a quelli
            %calcolati in precedenza
            NParams = NParams+prod(W1)+prod(B1)+prod(W2)+prod(B2);
        end
end