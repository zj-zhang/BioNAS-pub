'''
Use generator to train a CNN model for
all eCLIP sites
Zijun Zhang
7.18.2018
'''

from cnnModel import *
from makeTrainData import *


GENOME_FN = '/home/zzj/scratch/hg19/hg19.noRand.fa'
#NUM_TRAIN_SAMPLES = 1277907
#NUM_TRAIN_SAMPLES = 334327
NUM_TRAIN_SAMPLES = 538956
#NUM_VAL_SAMPLES = 141990
#NUM_VAL_SAMPLES = 37147
NUM_VAL_SAMPLES = 59884
BATCH_SIZE = 500


# RBFOX2
#Y_IDX = [209, 210, 211, 212]
Y_IDX = [197, 198, 199, 200]

def calculating_class_weights(y_true):
	from sklearn.utils.class_weight import compute_class_weight
	number_dim = np.shape(y_true)[1]
	weights = np.empty([number_dim, 2])
	for i in range(number_dim):
		weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
	#upper_bound = np.percentile(weights, 90)
	upper_bound = 10.
	weights[np.where(weights>upper_bound)] = upper_bound
	return weights

print("preparing generator..")
train_label = read_label('label_matrix.train.bed.gz')
neg_idx = np.where(train_label.iloc[:,Y_IDX[0]]==0)[0]
train_label.drop(
	train_label.index[np.random.choice(
		neg_idx,
		int(len(neg_idx)-
			np.sum(train_label.iloc[:,Y_IDX[0]])*1), 
		replace=False)],
	axis=0,
	inplace=True)
NUM_TRAIN_SAMPLES = train_label.shape[0]
train_gen = get_generator(train_label, batch_size=BATCH_SIZE,
	genome_fn=GENOME_FN,
	y_idx=Y_IDX)

print("reading val data..")
val_label = read_label('label_matrix.val.bed.gz')
val_gen = get_generator(val_label, batch_size=NUM_VAL_SAMPLES,
	genome_fn=GENOME_FN,
	y_idx=Y_IDX)
x_val, y_val = next(val_gen)
estim_class_weights = calculating_class_weights(y_val)

print("training model..")
model = build_model(num_output=len(Y_IDX), class_weights=estim_class_weights)

checkpointer = ModelCheckpoint(filepath="bestmodel.h5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
#predict_history = Predict_History()


history = model.fit_generator(train_gen, 
	steps_per_epoch=NUM_TRAIN_SAMPLES//BATCH_SIZE,
	epochs=200, 
	verbose=1,
	validation_data=(x_val, y_val),
	callbacks=[checkpointer, earlystopper],
	use_multiprocessing=True, 
	max_queue_size=10)