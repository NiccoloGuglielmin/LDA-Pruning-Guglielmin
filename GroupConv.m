classdef GroupConv < nnet.layer.Layer & nnet.layer.Formattable %(Optional) 
    
    properties %Proprietà del layer
        OriginalLayer;
    end

    properties (Learnable) %Parametri learnable del layer
        Net1;
        Net2;
    end
    
    methods
        
        %Costruttore
        function self = GroupConv(ConvolutionLayer,Weights,Bias)
            
            self.OriginalLayer = ConvolutionLayer;
            self.Name = ConvolutionLayer.Name;
            
            w = Weights{1};
            b = Bias{1};
            s = size(w);
            newLayer = convolution2dLayer(ConvolutionLayer.FilterSize,s(4),...
                'Name',strcat(ConvolutionLayer.Name,'g1'),...
                'Weights',w,...
                'Bias',b,...
                'Stride',ConvolutionLayer.Stride,...
                'DilationFactor',ConvolutionLayer.DilationFactor,...
                'Padding',ConvolutionLayer.PaddingSize,...
                'PaddingValue',ConvolutionLayer.PaddingValue);
            
            lgraph = layerGraph(newLayer);
            self.Net1 = dlnetwork(lgraph,'Initialize',false);
            
            w = Weights{2};
            b = Bias{2};
            s = size(w);
            newLayer = convolution2dLayer(ConvolutionLayer.FilterSize,s(4),...
                'Name',strcat(ConvolutionLayer.Name,'g2'),...
                'Weights',w,...
                'Bias',b,...
                'Stride',ConvolutionLayer.Stride,...
                'DilationFactor',ConvolutionLayer.DilationFactor,...
                'Padding',ConvolutionLayer.PaddingSize,...
                'PaddingValue',ConvolutionLayer.PaddingValue);
            lgraph = layerGraph(newLayer);
            self.Net2 = dlnetwork(lgraph,'Initialize',false);
            
        end
        
        %Funzione forward per la prediction del layer:
        %svolge un'operazione di forward dei dati di input attraverso il 
        %layer al momento della prediction ed emette in output il risultato
        function Z = predict(self,X)
            
            totChannels = 0;
            
            layer = self.Net1.Layers(1);
            s = size(layer.Weights);
            s1 = size(X);
            Xi = X(:,:,totChannels+1:totChannels+s(3),:);
            Z1 = self.Net1.predict(Xi);
            
            totChannels = totChannels+s(3);
            
            layer = self.Net2.Layers(1);
            s = size(layer.Weights);
            Xi = X(:,:,totChannels+1:totChannels+s(3),:);
            Z2 = self.Net2.predict(Xi);
            
            Z = cat(3,Z1,Z2);
            
        end
        
    end
    
end