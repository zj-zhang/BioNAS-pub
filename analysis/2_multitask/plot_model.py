from keras.utils import plot_model
from keras.models import load_model

from pkg_resources import resource_filename
from BioNAS.KFunctions.K_func import *
from BioNAS.utils.motif import draw_dnalogo_matplot, draw_dnalogo_Rscript


# load model
h_mod = load_model('high_kn_model.h5')
#m_mod = load_model('mid_kn_model.h5')
#l_mod = load_model('low_kn_model.h5')

# plot model
plot_model(h_mod, to_file="high_kn_model.pdf")


mkf = Motif_K_function(temperature=0.1, Lambda_regularizer=0.01)
mkf.knowledge_encoder(['CTCF_known1', 'IRF_known1'], resource_filename('BioNAS.resources', 'rbp_motif/encode_motifs.txt.gz'), False)

draw_dnalogo_Rscript(mkf.W_knowledge['CTCF_known1'].T, 'CTCF.pdf')
draw_dnalogo_Rscript(mkf.W_knowledge['IRF_known1'].T, 'IRF.pdf')
