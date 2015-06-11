% Run quadratic program for determining best order of features
function EEG_quadprog_ICA
    matFile = 'EEG_Mat/AH04_1.mat';
    % Read the feature vectors for all sets
    % disp([core_patients{idx} int2str(thisRecs(idx2))]);
    load(matFile, 'combFeatICA');
    tranSet = combFeatICA;
    % full data quad prog
    Qcorr = zeros(size(tranSet,2),size(tranSet,2));
    % generate correlation matrix
    for c1 = 1 : size(tranSet,2)
        for c2 = c1 : size(tranSet,2)
            CC = corr([tranSet(:,c1),tranSet(:,c2)]);
            Qcorr(c1,c2) = CC(1,2);
            Qcorr(c2,c1) = Qcorr(c1,c2);
        end
    end
    
    % create matrix of form b0 + b1*x1(t-1) + b2*x2(t-1) + ...
    X = tranSet(1:end-1,:); 
    Xfull = [X, ones(size(X,1),1)];
    Y = tranSet(2:end,:); % normalized feature vector

    % find coefficients of regression
    pX = pinv(Xfull)*Y; hPar = pX';

    % find column norms:
    colnorm = zeros(size(hPar,1),1);
    for cntr = 1 : size(hPar,1)
        colnorm(cntr) = norm(hPar(:,cntr));
    end

    % seizure data quad prog       
    cols = find(isnan(Qcorr));
    for cntr = 1 : size(cols,1)
        Qcorr(cols) = 0;
    end

    F_bestnorm = colnorm;
    F_bestnorm = F_bestnorm/max(F_bestnorm);
    A = mean(Qcorr(:))/(mean(Qcorr(:))+mean(F_bestnorm));
    n = size(tranSet,2);
    onev = ones(n,1); zerv = zeros(n,1);

    temp = quadprog(Qcorr*(1-A),-F_bestnorm*A,[],[],onev',1,zerv,onev);
    [sorted_temp, perm_temp] = sort(temp, 'descend');
    [row,col] = find(sorted_temp < 10^-8);
    
    for cntr = 1 : size(row,1)
        sorted_temp(row(cntr),col(cntr)) = 0;
    end
    [row,col] = find(sorted_temp > 0);
    for cntr = 1 : size(row,1)
        disp([sorted_temp(row(cntr),col(cntr)) perm_temp(row(cntr),col(cntr))]);
    end
end
