'''
Use generator to train a CNN model for
all eCLIP sites
Zijun Zhang
7.18.2018
'''

from cnnModel import *
from makeTrainData import *


GENOME_FN = '/home/zijun/Workspace/hg19/hg19.noRand.fa'
#NUM_TRAIN_SAMPLES = 1277907
#NUM_TRAIN_SAMPLES = 334327
NUM_TRAIN_SAMPLES = 538956
#NUM_VAL_SAMPLES = 141990
#NUM_VAL_SAMPLES = 37147
NUM_VAL_SAMPLES = 59884
BATCH_SIZE = 100


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
train_gen = get_generator(train_label, batch_size=BATCH_SIZE,
	genome_fn=GENOME_FN)

print("reading val data..")
val_label = read_label('label_matrix.val.bed.gz')
val_gen = get_generator(val_label, batch_size=NUM_VAL_SAMPLES,
	genome_fn=GENOME_FN)
x_val, y_val = next(val_gen)
estim_class_weights = calculating_class_weights(y_val)

print("training model..")
#model = build_model(class_weights=estim_class_weights)
model = build_model()

checkpointer = ModelCheckpoint(filepath="bestmodel.h5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
#predict_history = Predict_History()


history = model.fit_generator(train_gen, 
	steps_per_epoch=NUM_TRAIN_SAMPLES//BATCH_SIZE,
	epochs=100, 
	verbose=1,
	validation_data=(x_val, y_val),
	callbacks=[checkpointer, earlystopper],
	use_multiprocessing=True, 
	max_queue_size=10)
