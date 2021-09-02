%{

clc %Elimina tutto il testo dalla Command Window
clear all %Elimina tutte le variabili contenute nel Workspace
close all %Chiude tutte le Figure aperte

%Caricamento della rete di partenza
load('NetAndData.mat','netTransfer');
n = netTransfer;

%}

%Conversione della rete da oggetto di tipo SeriesNetwork oppure LayerGraph 
%in un oggetto dlnetwork, senza includere il classificationLayer di output
function [dlnet] = modifyNet(n)

    %figure
    %plot(n)

    %Per coprire il caso in cui n è di tipo dlnetwork
    dlnet = n;
    
    %Rimozione dell'ultimo layer (classificationLayer)
    if isa(n,'SeriesNetwork')
        g = layerGraph(n.Layers(1:length(n.Layers)-1)); %Rimozione layer di output
        %Conversione della rete modificata in una rete dlnetwork
        dlnet = dlnetwork(g)
    else %Controllare come si comporta il tutto se n è di tipo dlnetwork (vedi prova.m) ==> errore
        if ~isa(n,'dlnetwork') %Per gestire anche il caso generale
            g = layerGraph(n);
            if isequal(class(g.Layers(length(g.Layers))),'nnet.cnn.layer.ClassificationOutputLayer')
                OutputLName = g.Layers(length(g.Layers)).Name;
                g = removeLayers(g,OutputLName); %Rimozione layer di output
            end
            %Conversione della rete modificata in una rete dlnetwork
            dlnet = dlnetwork(g)
        end
    end

    %figure
    %plot(g)

    %Conversione della rete modificata in una rete dlnetwork
    %dlnet = dlnetwork(g)

    %Salvataggio della rete convertita in dlnetwork (con rimozione del
    %classificationLayer di output)
    save('ModifiedNet','dlnet')
 
end