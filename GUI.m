function varargout = GUI(varargin)
% GUI MATLAB code for GUI.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI

% Last Modified by GUIDE v2.5 17-Jan-2022 11:34:44

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI is made visible.
function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI (see VARARGIN)

% Choose default command line output for GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in Dataset.
function Dataset_Callback(hObject, eventdata, handles)
% hObject    handle to Dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Dataset contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Dataset

dataset_index = get(handles.Dataset, 'Value');
switch dataset_index 
    case 1
        video_dir =[pwd '/Video_Dataset_1/videoFolder/']; 
        frame_dir =[pwd '/Video_Dataset_1/Frames/']; 
        data_dir = [pwd '/Video_Dataset_1/hashCodes/'];
        feature_dir = [pwd '/Video_Dataset_1/features/'];
        colorData = 0;   
        
    case 2
        video_dir =[pwd '/Video_Dataset_2/videoFolder/']; 
        frame_dir =[pwd '/Video_Dataset_1/Frames/']; 
        data_dir = [pwd '/Video_Dataset_2/hashCodes/'];
        feature_dir = [pwd '/Video_Dataset_2/features/'];
        colorData = 0;    
end
% if(colorData == 1)
% load([data_dir '/colorLabel']);
% handles.colorLabel = colorLabel;
% end


load([data_dir '/filenames']);
load([data_dir '/targets']);

handles.filenames = filenames;
handles.targets = targets;
handles.video_dir = video_dir;
handles.frame_dir = frame_dir;
handles.data_dir  = data_dir;
handles.feature_dir = feature_dir;

set(handles.QueryName1,'String', filenames);
set(handles.QueryName2,'String', filenames);
set(handles.QueryName3,'String', filenames);


guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function Dataset_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





% --- Executes on selection change in QueryName1.
function QueryName1_Callback(hObject, eventdata, handles)
% hObject    handle to QueryName1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns QueryName1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from QueryName1

imindex = get(hObject,'Value');
video_dir = handles.video_dir;
frame_dir = handles.frame_dir;
fname = [handles.filenames{imindex}];
frame = [frame_dir fname '/' fname '_2.jpg'];

axes(handles.axes1);
imshow(frame);  axis image;

handles.q1Idx = imindex;
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function QueryName1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to QueryName1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in QueryName2.
function QueryName2_Callback(hObject, eventdata, handles)
% hObject    handle to QueryName2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns QueryName2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from QueryName2

imindex = get(hObject,'Value');
video_dir = handles.video_dir;
frame_dir = handles.frame_dir;
fname = [handles.filenames{imindex}];
frame = [frame_dir fname '/' fname '_2.jpg'];

axes(handles.axes2);
imshow(frame);  axis image;

handles.q2Idx = imindex;
guidata(hObject, handles);




% --- Executes during object creation, after setting all properties.
function QueryName2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to QueryName2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes during object creation, after setting all properties.
function FrontSelector_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FrontSelector (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes during object creation, after setting all properties.
function ImageSelector_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ImageSelector (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function main_Callback(hObject, eventdata, handles)
% hObject    handle to main (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of main as text
%        str2double(get(hObject,'String')) returns contents of main as a double


% --- Executes during object creation, after setting all properties.
function main_CreateFcn(hObject, eventdata, handles)
% hObject    handle to main (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on selection change in hashCodeSelection_f.
function hashCodeSelection_f_Callback(hObject, eventdata, handles)
% hObject    handle to hashCodeSelection_f (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns hashCodeSelection_f contents as cell array
%        contents{get(hObject,'Value')} returns selected item from hashCodeSelection_f



filenames = handles.filenames;
targets   = handles.targets;
video_dir = handles.video_dir;
frame_dir = handles.frame_dir;
data_dir  = handles.data_dir;
feature_dir = handles.feature_dir;

% load([data_dir '/filenames']); % File names
% load([data_dir '/targets']);   % Labels

hashCode_index = get(handles.hashCodeSelection_f, 'Value');

switch hashCode_index
           
    case 1
        load([data_dir '/hashCodes_16']); 
        data = hashCodes_16;
        load([feature_dir '/features_16']); 
        features = features_16;
        %data = features_128 > 0.5;
    case 2
       load([data_dir '/hashCodes_32']); 
       data = hashCodes_32;
       load([feature_dir '/features_32']); 
       features = features_32;
       %data = features_256 > 0.5;
    case 3
        load([data_dir '/hashCodes_64']); 
        data = hashCodes_64;
        load([feature_dir '/features_64']); 
        features = features_64;
        %data = features_512 > 0.5;
    
end




% set(handles.QueryName1,'String', filenames);
% set(handles.QueryName2,'String', filenames);

%handles.filenames = filenames;
%handles.targets = targets;
handles.data = data;
handles.features = features;

guidata(hObject, handles);



% --- Executes during object creation, after setting all properties.
function hashCodeSelection_f_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hashCodeSelection_f (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes during object creation, after setting all properties.
function pushbutton14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on selection change in QueryName3.
function QueryName3_Callback(hObject, eventdata, handles)
% hObject    handle to QueryName3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns QueryName3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from QueryName3


imindex = get(hObject,'Value');
video_dir = handles.video_dir;
frame_dir = handles.frame_dir;
fname = [handles.filenames{imindex}];
frame = [frame_dir fname '/' fname '_2.jpg'];

axes(handles.axes37);
imshow(frame);  axis image;

handles.q3Idx = imindex;
guidata(hObject, handles);



% --- Executes during object creation, after setting all properties.
function QueryName3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to QueryName3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function pp_retrieved_Callback(hObject, eventdata, handles)
% hObject    handle to pp_retrieved (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pp_retrieved as text
%        str2double(get(hObject,'String')) returns contents of pp_retrieved as a double
pp_retrieved = str2double(get(handles.pp_retrieved,'string'));
handles.pp_retrieved = pp_retrieved;

% --- Executes during object creation, after setting all properties.
function pp_retrieved_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pp_retrieved (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in rank_euclid.
function rank_euclid_Callback(hObject, eventdata, handles)
% hObject    handle to rank_euclid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cla(handles.axes3,'reset');
P = str2double(get(handles.pp_retrieved, 'String'));

queryIndex1 = handles.q1Idx;
queryIndex2 = handles.q2Idx;
queryIndex3 = handles.q3Idx;

features = handles.features;
targets = handles.targets;
data = handles.data;
N = length(handles.filenames);

q_1 = data(queryIndex1,:); 
q_2 = data(queryIndex2,:); 
q_3 = data(queryIndex3,:); 

q1new = repmat(q_1,N,1);
q2new = repmat(q_2,N,1);
q3new = repmat(q_3,N,1);

dist_1 = xor(data, q1new);
dist_2 = xor(data, q2new);
dist_3 = xor(data, q3new);

hamming_dist1 = sum(dist_1,2);
hamming_dist2 = sum(dist_2,2);
hamming_dist3 = sum(dist_3,2);

n_hamming_dist1 = mat2gray(hamming_dist1);
n_hamming_dist2 = mat2gray(hamming_dist2);
n_hamming_dist3 = mat2gray(hamming_dist3);
 
 
X = zeros(3,N);
X(1,:) = hamming_dist1;
X(2,:) = hamming_dist2;
X(3,:) = hamming_dist3;
X = (X)';
input = unique(X, 'rows');

axes(handles.axes3);
hold off;
scatter3(X(:,1),X(:,2),X(:,3),'k.','MarkerFaceColor',[0 1 0])    
set(gca,'FontSize',10); hold on;
%scatter3(X(:,1),X(:,2), X(:,3),'.');
view(3); rotate3d on; 
hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
union_of_query_labels = or(targets(queryIndex1, :), targets(queryIndex2, : ));
union_of_query_labels = or(union_of_query_labels  , targets(queryIndex3, : ));
absolute_union_of_query_labels = nnz(union_of_query_labels);   
%{    
 for e = 1:N        
            MQUR_ALL(e,:) =  nnz( and(targets(e,:) , union_of_query_labels ) ) / absolute_union_of_query_labels ;
            
 end

MQUR_ONE = find(MQUR_ALL == 1);
scatter3(X(MQUR_ONE,1),X(MQUR_ONE,2),X(MQUR_ONE,3),'gs');
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[K,L] = size(unique(X,'rows'));  %% Number of unique pareto points 
set(handles.num_pp,'String',num2str(K))


% queries location and Optimum point location
q1 = X(queryIndex1,:); 
q2 = X(queryIndex2,:); 
q3 = X(queryIndex3,:); 

Cmn = (q1(:) + q2(:) + q3(:)).'/2;
plot3(q1 , q2, q3, 'p' , 'MarkerSize',20,'MarkerFaceColor',[1 0 1]);
plot3(Cmn(1), Cmn(2), Cmn(3), 'd', 'MarkerSize',20 , 'MarkerFaceColor',[1 0 0]);
%plot3(q1 , q2, q3, 'gd' , 'LineWidth', 20); 
%plot3(Cmn(1), Cmn(2), Cmn(3), 'ro' , 'LineWidth', 10);

Dissim = EuDist2(Cmn, input);
Dissim = Dissim';

[DissimSorted, DissimIndex] = sort(Dissim,'ascend'); % short & get indexes(before shorting)


%%%%%%%%%%%%% Choose First P Shortest distances %%%%%%%%%%%%%%%%%%%%%%%%

DissimIndex_P = DissimIndex(1:P, :); 
Retrieved_PP_indexes  =    ismember(X, input(DissimIndex_P,:),'rows'); 
Retrieved_Items       =    find(Retrieved_PP_indexes); 

scatter3(X(Retrieved_Items,1),X(Retrieved_Items,2),X(Retrieved_Items,3),'ys', 'LineWidth', 2); 
[x,y,z] = sphere;
radius = DissimSorted(P);
x = x * radius;
y = y * radius;
z = z * radius;
h = surfl(x + Cmn(1),y + Cmn(2),z + Cmn(3));
set(h, 'FaceAlpha', 0.1)
shading interp


%%%%%%%%%%%%%%%%%%%%  ReRanking, rearrange P items by features  %%%%%%%%%%% 

% Add queries to Feature Pareto space, for creating Cmn_f
Retrieved_Items(end+1,:) = queryIndex1;
Retrieved_Items(end+1,:) = queryIndex2;
Retrieved_Items(end+1,:) = queryIndex3;

rtr_idx2_features = features(Retrieved_Items(:,1), :);

f1 = features(queryIndex1,:); 
f2 = features(queryIndex2,:);
f3 = features(queryIndex3,:);

dist_f1 = pdist2(f1 , rtr_idx2_features , 'euclid' );
dist_f2 = pdist2(f2 , rtr_idx2_features , 'euclid' );
dist_f3 = pdist2(f3 , rtr_idx2_features , 'euclid' );

[M2,B] = size(rtr_idx2_features(:,1));
YY = zeros(3,M2);
YY(1,:) = dist_f1;
YY(2,:) = dist_f2;
YY(3,:) = dist_f3;
YY = (YY)';

qf1 = YY(end-2,:); 
qf2 = YY(end-1,:); 
qf3 = YY(end,:);

Cmn_f = (qf1(:) + qf2(:) + qf3(:)).'/2;
plot3(Cmn_f(1), Cmn_f(2), Cmn_f(3), 's' , 'LineWidth', 2);

Dissim_f                = EuDist2(YY, Cmn_f, 'euclid');
Dissim_ff               = [Retrieved_Items, Dissim_f];
DissimSorted_f          = sortrows(Dissim_ff, 2);

Retrieved_Items_Ranked  = DissimSorted_f(:,1);

% Now remove last three rows (qf1&qf2&qf3) from Retrieved items
Retrieved_Items_Ranked(end) = [];
Retrieved_Items_Ranked(end) = [];
Retrieved_Items_Ranked(end) = [];


%%%%%%%%%%%%%%%%%%%%%%% Metrics & MQUR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

predicted_labels  = targets(Retrieved_Items_Ranked,:);

[num_R, ~]  = size(Retrieved_Items_Ranked);     
 for e = 1:num_R         
     MQUR_Ranked(e,:) =  nnz( and(predicted_labels(e,:) , union_of_query_labels ) ) / absolute_union_of_query_labels;       
                        
 end
      
for ff = 1:num_R
    if  MQUR_Ranked(ff,:) ~= 1   %%%%%% MQUR !DEN FARKLI OLANLARI SEÇ 
        Retrieved_Items_Ranked(ff,:) = 0;   
        MQUR_Ranked(ff,:) = 0;
    end
end

Retrieved_Items_Ranked( all(~Retrieved_Items_Ranked,2), : ) = [];
MQUR_Ranked( all(~MQUR_Ranked,2), : ) = [];
scatter3(X(Retrieved_Items_Ranked,1),X(Retrieved_Items_Ranked,2),X(Retrieved_Items_Ranked,3),'bs','LineWidth', 2);
 
 
predicted_labels_ranked  = targets(Retrieved_Items_Ranked,:);
%diff = ismember( predicted_labels_ranked, union_of_query_labels , 'rows' );
if isempty(MQUR_Ranked)
   MQUR_Ranked = 0;
end
  
num_nz = nnz( MQUR_Ranked(:,1) );
s = size(MQUR_Ranked(:,1), 1);
    
for j=1:s;        
    %Cummulative sum of the true-positive elements
    CUMM = cumsum(MQUR_Ranked);          
    Precision_AT_K(j,1) = ( CUMM(j,1)  ) ./ j;              
    Recall_AT_K(j,1) = ( CUMM(j,1)  ) ./ (num_nz); %              
end

avg_Precision = sum(Precision_AT_K(:,1) .* MQUR_Ranked(:,1) ) / num_nz;
avg_Precision(isnan(avg_Precision))=0;
% avg_Precision_OLD = sum(Precision_AT_K(:,1) ) / s;
 acc = num_nz / s;   % accuracy of the best cluster 


set(handles.acc_box,'String',num2str(acc));
set(handles.avg_prec,'String',num2str(avg_Precision));

          
      
      axes(handles.axes11);
      hold off;
      %plot(Recall_AT_K, Precision_AT_K);
      x = linspace(0,s);
      plot( Precision_AT_K )
      ylabel('Precision@k' ,'FontSize', 12)
      xlabel('Number of Rterieved Items' ,'FontSize', 12) 
      hold on;
         

         cla(handles.axes13,'reset');
         cla(handles.axes14,'reset');
         cla(handles.axes15,'reset');
         cla(handles.axes16,'reset');
         
         
                
         
        axes(handles.axes13);
        fname = [handles.filenames{Retrieved_Items_Ranked(1,1)}];
        frame = [handles.frame_dir fname '/' fname '_2.jpg'];
        imshow(imread(frame)); 
        set(handles.edit19,'string',num2str( handles.filenames{Retrieved_Items_Ranked(1,1)}));
        axis image
      
        axes(handles.axes14);
        fname = [handles.filenames{Retrieved_Items_Ranked(2,1)}];
        frame = [handles.frame_dir fname '/' fname '_2.jpg'];
        imshow(imread(frame)); 
        set(handles.edit20,'string',num2str(  handles.filenames{Retrieved_Items_Ranked(2,1)}));
        axis image
        
        axes(handles.axes15);
        fname = [handles.filenames{Retrieved_Items_Ranked(3,1)}];
        frame = [handles.frame_dir fname '/' fname '_2.jpg'];
        imshow(imread(frame)); 
        set(handles.edit21,'string',num2str( handles.filenames{Retrieved_Items_Ranked(3,1)}));
        axis image
        
        axes(handles.axes16);
        fname = [handles.filenames{Retrieved_Items_Ranked(4,1)}];
        frame = [handles.frame_dir fname '/' fname '_2.jpg'];
        imshow(imread(frame)); 
        set(handles.edit22,'string',num2str( handles.filenames{Retrieved_Items_Ranked(4,1)}));
        axis image
        
      
                
     
                          
      
 guidata(hObject, handles);


 



function edit23_Callback(hObject, eventdata, handles)
% hObject    handle to edit23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit23 as text
%        str2double(get(hObject,'String')) returns contents of edit23 as a double


% --- Executes during object creation, after setting all properties.
function edit23_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit24_Callback(hObject, eventdata, handles)
% hObject    handle to edit24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit24 as text
%        str2double(get(hObject,'String')) returns contents of edit24 as a double


% --- Executes during object creation, after setting all properties.
function edit24_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit25_Callback(hObject, eventdata, handles)
% hObject    handle to edit25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit25 as text
%        str2double(get(hObject,'String')) returns contents of edit25 as a double


% --- Executes during object creation, after setting all properties.
function edit25_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit26_Callback(hObject, eventdata, handles)
% hObject    handle to edit26 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit26 as text
%        str2double(get(hObject,'String')) returns contents of edit26 as a double


% --- Executes during object creation, after setting all properties.
function edit26_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit26 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit19_Callback(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit19 as text
%        str2double(get(hObject,'String')) returns contents of edit19 as a double


% --- Executes during object creation, after setting all properties.
function edit19_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit20_Callback(hObject, eventdata, handles)
% hObject    handle to edit20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit20 as text
%        str2double(get(hObject,'String')) returns contents of edit20 as a double


% --- Executes during object creation, after setting all properties.
function edit20_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit21_Callback(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit21 as text
%        str2double(get(hObject,'String')) returns contents of edit21 as a double


% --- Executes during object creation, after setting all properties.
function edit21_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit22_Callback(hObject, eventdata, handles)
% hObject    handle to edit22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit22 as text
%        str2double(get(hObject,'String')) returns contents of edit22 as a double


% --- Executes during object creation, after setting all properties.
function edit22_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function num_pp_Callback(hObject, eventdata, handles)
% hObject    handle to num_pp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of num_pp as text
%        str2double(get(hObject,'String')) returns contents of num_pp as a double


% --- Executes during object creation, after setting all properties.
function num_pp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to num_pp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function acc_box_Callback(hObject, eventdata, handles)
% hObject    handle to acc_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of acc_box as text
%        str2double(get(hObject,'String')) returns contents of acc_box as a double


% --- Executes during object creation, after setting all properties.
function acc_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to acc_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function avg_prec_Callback(hObject, eventdata, handles)
% hObject    handle to avg_prec (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of avg_prec as text
%        str2double(get(hObject,'String')) returns contents of avg_prec as a double


% --- Executes during object creation, after setting all properties.
function avg_prec_CreateFcn(hObject, eventdata, handles)
% hObject    handle to avg_prec (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
