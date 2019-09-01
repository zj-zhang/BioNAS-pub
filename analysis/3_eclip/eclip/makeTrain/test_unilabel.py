from sklearn import metrics
from cnnModel import *
from makeTrainData import *

#NUM_TESTING_SAMPLE = 55417
#NUM_TESTING_SAMPLE = 12385
NUM_TESTING_SAMPLE = 21229

# RBFOX2
#Y_IDX = [209, 210, 211, 212]
Y_IDX = [197, 198, 199, 200]



model = build_model(num_output=len(Y_IDX))
model.load_weights('bestmodel.h5')

test_label = read_label('label_matrix.testing_chr8.bed.gz')
test_gen = get_generator(test_label, genome_fn='/home/zzj/scratch/hg19/hg19.noRand.fa',
	batch_size=NUM_TESTING_SAMPLE,
	y_idx=Y_IDX)
test_label = test_label.iloc[:,Y_IDX]
label_colnames = test_label.columns.values



x_test, y_test = next(test_gen)
print(model.evaluate(x_test, y_test))


y_hat = model.predict(x_test)

with open('test_eval.txt', 'w') as fo:
	for i in range(y_test.shape[1]):
		try:
			auroc = metrics.roc_auc_score(y_test[:,i], y_hat[:,i])
			aupr = metrics.average_precision_score(y_test[:,i], y_hat[:,i])
			fo.write("{}\t{}\t{}\t{}\t{}\n".format(
				label_colnames[i],
				np.sum(y_test[:,i]),
				np.sum(y_test[:,i]==0),
				auroc,
				aupr
				)
			)
		except ValueError:
			pass