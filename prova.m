load('NetAndData.mat','netTransfer');
n = netTransfer;

dlnet = dlnetwork(n)

g = layerGraph(dlnet);

figure
plot(g)

%{

Error using dlnetwork (line 184)
Input must be a LayerGraph object or an array of layers.

Error in prova (line 4)
dlnet = dlnetwork(n)

%}