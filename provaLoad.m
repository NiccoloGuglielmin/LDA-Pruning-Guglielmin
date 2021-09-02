load('NetAndData.mat','netTransfer');
        n = netTransfer;
        %g = layerGraph(n.Layers(1:length(n.Layers)));
        %disp("here");analyzeNetwork(g);
        %g = layerGraph(n.Layers(1:length(n.Layers)-1));
        %dlnet = dlnetwork(g);disp("here");analyzeNetwork(dlnet);
        
        
        g = layerGraph(n.Layers(1:length(n.Layers)));
        dlnet = dlnetwork(g);
        disp("here");disp(dlnet.Layers(1:length(n.Layers)));