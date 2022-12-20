
function generate_csv(PassengerId, predicted_labels,filename)
% to round the probabilities to 0 and 1 i.e.0:not survived, 1:survived
Survived = round(predicted_labels);
out = table(PassengerId, Survived);
writetable(out,filename)
end