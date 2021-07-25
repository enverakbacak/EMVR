
% https://www.mathworks.com/matlabcentral/answers/361878-getting-the-data-points-of-each-cluster-in-kmeans
close all;
clear all;
clc;


load('Video_Dataset_3/hashCodes/hashCodes_64.mat');
data = hashCodes_64;
%data = max(data,0);
load('Video_Dataset_3/features/features_64.mat');
features = features_64;
load('Video_Dataset_3/hashCodes/targets.mat');
targets = targets;
load('Video_Dataset_3/hashCodes/filenames.mat');
queryIndex = xlsread('Video_Dataset_3/qGroups_4d.xls');
filenames = filenames;
N = length(filenames);
%}


queryIndex = transpose( queryIndex ); 
queryIndex1 = queryIndex(1,:);        % First element of Query Triplets
queryIndex2 = queryIndex(2,:);        % Second element of Query Triplets
queryIndex3 = queryIndex(3,:); 
queryIndex4 = queryIndex(4,:); 



 for l = 1:100 % Number of Query Pairs , CAN TRY FOR DIFFERENT HAMMING RADIUS ALSO ?????? how?
              
       q_1 = data(queryIndex1,:);         % q1 & q2 are query pairs in the loop
       q_2 = data(queryIndex2,:);
       q_3 = data(queryIndex3,:);
       q_4 = data(queryIndex4,:);
       
       q1_rep{l,:} = repmat(q_1(l,:),N,1); % Make query matrix size to the same as data matrix size
       q2_rep{l,:} = repmat(q_2(l,:),N,1);  
       q3_rep{l,:} = repmat(q_3(l,:),N,1); 
       q4_rep{l,:} = repmat(q_4(l,:),N,1); 
       xor_data_q1new{l,:} = xor(data, q1_rep{l,:}); % xor of data and query matrices
       xor_data_q2new{l,:} = xor(data, q2_rep{l,:}); 
       xor_data_q3new{l,:} = xor(data, q3_rep{l,:});
       xor_data_q4new{l,:} = xor(data, q4_rep{l,:});
       hamming_dist1{l,:} = sum(xor_data_q1new{l,:},2); % sum up rows to get hamming distances
       hamming_dist2{l,:} = sum(xor_data_q2new{l,:},2);
       hamming_dist3{l,:} = sum(xor_data_q3new{l,:},2);
       hamming_dist4{l,:} = sum(xor_data_q4new{l,:},2);
       %norm_hamming_dist1{l,:} =  hamming_dist1{l,:} / max( hamming_dist1{l,:}(:) ); % Normalize hamming  distances between 0&1
       %norm_hamming_dist2{l,:} =  hamming_dist2{l,:} / max( hamming_dist2{l,:}(:) );
       %dist1{l,:} = mat2gray(dist1{l,:}); % Normalize hamming  distances between 0&1
       %dist2{l,:} = mat2gray(dist2{l,:});     
        
       X{l,:} = zeros(4,N);
       X{l,:}(1,:) = hamming_dist1{l,:};
       X{l,:}(2,:) = hamming_dist2{l,:};    
       X{l,:}(3,:) = hamming_dist3{l,:};
       X{l,:}(4,:) = hamming_dist4{l,:};   
       X{l,:} = (X{l,:})';    
       input{l,:} = unique(X{l,:}, 'rows');
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
        union_of_query_labels{l,:} = or(targets(queryIndex(1,l), :), targets(queryIndex(2,l), : )); 
        union_of_query_labels{l,:} = or(union_of_query_labels{l,:},  targets(queryIndex(3,l), : )); 
        union_of_query_labels{l,:} = or(union_of_query_labels{l,:},  targets(queryIndex(4,l), : ));   
        absolute_union_of_query_labels{l,:} = nnz(union_of_query_labels{l,:} );
       
          
        for e = 1:N        
            MQUR_ALL{l,:}(e,:) =  nnz( and(targets(e,:) , union_of_query_labels{l,:} ) ) / absolute_union_of_query_labels{l,:} ;
            
        end

        % MQUR_ONE{l,:} = find(MQUR_ALL{l,:} == 1);
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
       
    
       %************* Cmn: Optimum Point *********
       % queries location 
       q1 = X{l,:}(queryIndex1,:); 
       q2 = X{l,:}(queryIndex2,:); 
       q3 = X{l,:}(queryIndex3,:);
       q4 = X{l,:}(queryIndex4,:);
    
       Cmn(l,:) = (q1(l,:) + q2(l,:) + q3(l,:) + q4(l,:)).'/2;
       Dissim{l,:} = EuDist2(Cmn(l,:), input{l,:});
       Dissim{l,:} = (Dissim{l,:})';

       [DissimSorted , DissimIndex ] = sort(Dissim{l,:}(:),'ascend'); % short & get indexes(before shorting)
 


       %%%%%%%%%%%%% Choose First P Shortest distances %%%%%%%%%%%%%%%%%%%%%%%%       
       P = 20;
       DissimIndex_P                      =    DissimIndex(1:P, :); 
       Retrieved_PP_indexes{l,:}          =    ismember(X{l,:}, input{l,:}(DissimIndex_P,:),'rows'); 
       Retrieved_Items{l,:}               =    find(Retrieved_PP_indexes{l,:} ); 
       
       %%%%%%%%%%%%%%%%%%%%  ReRanking, rearrange P items by features  %%%%%%%%%%% 

        % Add queries to Feature Pareto space, for creating Cmn_f
        Retrieved_Items{l,:}(end+1,:) = queryIndex1(:,l);
        Retrieved_Items{l,:}(end+1,:) = queryIndex2(:,l);
        Retrieved_Items{l,:}(end+1,:) = queryIndex3(:,l);
        Retrieved_Items{l,:}(end+1,:) = queryIndex4(:,l);
    
    
        rtr_idx2_features = features(Retrieved_Items{l,:}, :);
 
        % features of query pairs
        f1 = features(queryIndex1,:); 
        f2 = features(queryIndex2,:); 
        f3 = features(queryIndex3,:); 
        f4 = features(queryIndex4,:); 
    
        % Distance from each query pair features to retrieved items 
        dist_f1{l,:} = pdist2(f1(l,:) , rtr_idx2_features , 'euclid' );
        dist_f2{l,:} = pdist2(f2(l,:) , rtr_idx2_features , 'euclid' ); 
        dist_f3{l,:} = pdist2(f3(l,:) , rtr_idx2_features , 'euclid' ); 
        dist_f4{l,:} = pdist2(f4(l,:) , rtr_idx2_features , 'euclid' );
         
        % How many rows of trr_idx2
        [ M(l,:), ~] = size(rtr_idx2_features);

        % Create  4xM zero vector for assigne each distance (dis_f1,f2, f3) to them
        % YY is Pareto space formed by cnn features of retrived itmes,
        % which are retrieved by has codes
        YY{l,:} = zeros(4,M(l,:)); 
        YY{l,:}(1,:) = dist_f1{l,:};
        YY{l,:}(2,:) = dist_f2{l,:};
        YY{l,:}(3,:) = dist_f3{l,:};
        YY{l,:}(4,:) = dist_f4{l,:};
        YY{l,:} = (YY{l,:})';
 
        qf1(l,:) = YY{l,:}(end-3,:); 
        qf2(l,:) = YY{l,:}(end-2,:);
        qf3(l,:) = YY{l,:}(end-1,:);
        qf4(l,:) = YY{l,:}(end,:);
    
        % Find distance between each query for query pairs
        % Optimum Point of Pareto space formed by features of the retrived items
        %df(l,:) = pdist2(qf1(l,:),qf2(l,:),'euclid' );
        %Cmn_f(l,:) = [df(l,:)/2 , df(l,:)/2];
        Cmn_f(l,:) = (qf1(l,:) + qf2(l,:) + qf3(l,:) + + qf4(l,:)).'/2;
    
    
    
        % DD is the distance between each retrieved items to optimum point
        Dissim_f{l,:}                = EuDist2(YY{l,:}, Cmn_f(l,:), 'euclid');
        Dissim_ff{l,:}               = [Retrieved_Items{l,:}, Dissim_f{l,:}];
        DissimSorted_f{l,:}          = sortrows(Dissim_ff{l,:}, 2);

        Retrieved_Items_Ranked{l,:}  = DissimSorted_f{l,:}(:,1);

        % Now remove last three rows (qf1&qf2&qf3) from Retrieved items
        Retrieved_Items_Ranked{l,:}(end) = [];
        Retrieved_Items_Ranked{l,:}(end) = [];
        Retrieved_Items_Ranked{l,:}(end) = [];
        Retrieved_Items_Ranked{l,:}(end) = [];
        
        [num_R2(l,:), ~]  = size(Retrieved_Items_Ranked{l,:} );
     
              
        Retrieved_Items_Ranked_MQUR_scores{l,:} = MQUR_ALL{l,:}(Retrieved_Items_Ranked{l,:}, :);
   
     
        nDCG_EMIR{l,:}(1,:) = Retrieved_Items_Ranked_MQUR_scores{l,:}(1,:); 
        for a=2:num_R2(l,:)
            nDCG_EMIR{l,:}(:,a) =  Retrieved_Items_Ranked_MQUR_scores{l,:}(a,:) / log2(a);
        end       
    
 end
 

 
 
 % Zero padding & mean of nDCG score
 
for ll=1:l    
        [~ , C_1(ll,:)] = size(nDCG_EMIR{ll,:}(:,:)); % Find size of each matrix in the n_rigth_DCG array   
end
max_1 = max(C_1(:)); % Find max size of nDCG_EMIR row


for ll=1:l
       nDCG_EMIR_zp{ll,:}(1,:) =  [ nDCG_EMIR{ll,:}(1,:) ,(zeros(1 ,max_1 - C_1(ll,:) ))]; % Zero padding of all elements  in the n_rigth_DCG
end

nDCG_EMIR_zp_sum = 0;
for ll=1:l
       nDCG_EMIR_zp_sum = nDCG_EMIR_zp_sum + nDCG_EMIR_zp{ll,:}(1,:);
end
nDCG_EMIR_mean =  nDCG_EMIR_zp_sum/ll;

n_DCG_mean = nDCG_EMIR_mean;
plot(n_DCG_mean); 
ylabel('nDCG scores of EMIR')
xlabel('Number of Retrived Items') 



