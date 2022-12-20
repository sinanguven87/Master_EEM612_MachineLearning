%% Verilerin Matlab Ortamına Aktarımı
% Çalışma alanı temizle
clc;
clear;
close all;
% Eğitim ve test verisi oku
Train = readtable('train.csv','Format','%f%f%f%q%C%f%f%f%q%f%q%C');
Test = readtable('test.csv','Format','%f%f%q%C%f%f%f%q%f%q%C');

%% Veri Yönetimi
%Test matrisinden yolcu listesinin alinmasi
PassangerIdList=Test.PassengerId;
%Olmayan degerleri ortalama deger ile degistir.
avgAge = mean(Train.Age,'omitnan');
Train.Age(isnan(Train.Age)) = avgAge;  
Test.Age(isnan(Test.Age)) = avgAge;    

%Pclass değerlerinden yola çıkarak kayıp olan fare değerlerinin tahmini
%Burada fare değerleri tahmin edilirken ortalama sınıf değerlerine göre
%fare ataması yapılmıştır.
fare = grpstats(Train(:,{'Pclass','Fare'}),'Pclass');   % sınıf ortalama değerini hesapla
figure;
plot(fare.Pclass,fare.mean_Fare);
title('Müşteri sınıfı-Ortalama bilet ücreti grafiği');
xlabel('Müşteri Sınıfı')
ylabel('Ortalama bilet ücreti')
grid on
for i = 1:height(fare) 
    Train.Fare(Train.Pclass == i & isnan(Train.Fare)) = fare.mean_Fare(i);
    Test.Fare(Test.Pclass == i & isnan(Test.Fare)) = fare.mean_Fare(i);
end


%Olmayan embarkled değerlerini daha sık olan S ile değiştir.
histogram(Train.Embarked);
ylabel("Embarked");
title("Embarked");
legend("show");
% En fazla Embarked değerini bul
freqVal = mode(Train.Embarked);

% Embarked değeri boş kısımlara en sık değeri uygula
Train.Embarked(isundefined(Train.Embarked)) = freqVal;
Test.Embarked(isundefined(Test.Embarked)) = freqVal;

% Kategorik değerden sayısal değere çevir
Train.Embarked = double(Train.Embarked);
Test.Embarked = double(Test.Embarked);

%Cinsiyet değerlerini sayısal değere çevir
Train.Sex = double(Train.Sex);
Test.Sex = double(Test.Sex);


% Yaş değerlerini gruplandır
Train.Age = double(discretize(Train.Age, [0:10:20 65 80], ...
    'categorical',{'cocuk','genc','ortayas','yasli'}));
Test.Age = double(discretize(Test.Age, [0:10:20 65 80], ...
    'categorical',{'cocuk','genc','ortayas','yasli'}));

% Fare değerlerini gruplandır.
Train.Fare = double(discretize(Train.Fare, [0:10:30, 100, 520], ...
    'categorical',{'<10','10-20','20-30','30-100','>100'}));
Test.Fare = double(discretize(Test.Fare, [0:10:30, 100, 520], ...
    'categorical',{'<10','10-20','20-30','30-100','>100'}));

%Name, PassangerId,Ticket ve Cabin öznitelikleri kullanılamayacak
%durumdadır.
Train(:,{'Name','PassengerId','Ticket','Cabin'}) = [];
Test(:,{'Name','PassengerId','Ticket','Cabin'}) = [];


%% MantıksaL Regression işlemi
tbl=Train(:,2:8);
tbl.Survived=Train.Survived;
reg_model = fitglm(tbl, 'Distribution','binomial');
ypred = predict(reg_model,tbl(:,1:end-1));
ypred = round(ypred); %Olasılıklar 0-1' yuvarlanır.
Confusion_Matrix = confusionmat(tbl.Survived,ypred);
AccuracyReg = trace(Confusion_Matrix)/sum(Confusion_Matrix, 'all');

%%Karar Ağacı
% Karar ağacı oluşturulur ve çizdirilir.
mytree = fitctree(Train(:,2:end),Train(:,1));
view(mytree, 'Mode', 'graph')

%Karar ağacında çapraz validasyon hataları gidermek için en uygun budama
%seviyesi belirlenir.Aynı zamanda gereksiz seviyeler budanır.
[~,~,~,BestLevel] = cvloss(mytree,'subtrees','all','treesize','min');
prunetree = prune(mytree,'Level',BestLevel);
view(prunetree,'mode','graph')

% Karar ağacı doğruluk değeri bulunur.
label = predict(prunetree,Train(:,2:end));
Confusion_Matrix_Tree = confusionmat(Train.Survived,label);
Accuracy_Tree = trace(Confusion_Matrix_Tree)/sum(Confusion_Matrix_Tree, 'all');

%%KNN komşuluk
%Optimum komşuluk sayısını bulabilmek için komşuluk değerlerini 1:20 arasında değiştir.
cv_loss_knn=zeros(20,1);
Neighbors= transpose(1:20);
for num_neighbours = 1:20
rng(1);
knn_model = ClassificationKNN.fit(Train(:,2:8),Train(:,1), 'NumNeighbors',num_neighbours);
cvc_model = crossval(knn_model);
cv_loss = kfoldLoss(cvc_model);
cv_loss_knn(num_neighbours)=cv_loss;
end

% Knn değerlerini çizdir.
figure('Name', 'KNN')
plot(Neighbors, cv_loss_knn, 'LineWidth',1.5)
title('Knn komşuluğuna göre çağraz validasyon değeri', 'FontSize',14);
xlabel('Komşuluk sayısı', 'FontSize',14);
ylabel('Loss', 'FontSize',14);

% Optimal komşuluk sayısı için Knn modelini elde et
[M,minimumIndex] = min(cv_loss_knn); 
knn_model = ClassificationKNN.fit(Train(:,2:end),Train(:,1), 'NumNeighbors',minimumIndex);

% Doğruluk değerini bul.
label_knn = predict(knn_model,Train(:,2:end));
Confusion_Matrix_knn = confusionmat(Train.Survived,label_knn);
Accuracy_knn = trace(Confusion_Matrix_knn)/sum(Confusion_Matrix_knn, 'all');
  




%% 
%Train işlemi yapılan modellerle test işlemi başlat



%Regresyon modeli çıktılarını csv olarak yazdır
reg_modelTest = predict(reg_model,Test(:,1:end));
generate_csv(PassangerIdList, reg_modelTest,'Regression.csv');

%Tree modeli çıktılarını csv olarak yazdır
prunetreeTest = predict(prunetree,Test(:,1:end));
generate_csv(PassangerIdList, prunetreeTest,'Tree.csv');
    
%Knn modeli çıktılarını csv olarak yazdır
knn_modelTest = predict(knn_model,Test(:,1:end));
generate_csv(PassangerIdList, knn_modelTest,'Knn.csv');



