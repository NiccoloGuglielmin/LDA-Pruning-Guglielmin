%Model Gradients Function: funzione che riceve in input un oggetto dlnetwork,
%un mini-batch di dati di input dlX con le label corrispondenti Y e ritorna
%i gradienti della funzione di perdita rispetto ai parametri learnable in 
%dlnet, dello state della rete e della loss.
%Per calcolare automaticamente i gradienti viene utilizzata la funzione dlgradient
function [gradients,state,loss] = modelGradient(dlnet,dlX,Y)

    %Ritorna l'output dlY della rete dato in input dlX durante il training
    [dlYPred,state] = forward(dlnet,dlX);

    %L'operazione di cross-entropy calcola la loss tra la prediction della
    %rete e i valori target per compiti di single e multi-label.
    %Ritorna la loss cross-entropy tra l'array formattato dlarray e Y che
    %contiene le prediction ed i valori target per compiti di
    %classificazione single-label. L'output (loss) è un dlarray di scalari
    %non formattato
    loss = crossentropy(dlYPred,Y);
    %Ritorna i gradienti della variabile loss rispetto alla variabile dlnet.Learnables
    gradients = dlgradient(loss,dlnet.Learnables);

    %Conversione del tipo dei dati
    %loss=cast(extractdata(loss),'double');
    loss = double(gather(extractdata(loss)));

end