import bbknn
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
sc.settings.verbosity = 3			 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi_save=800, dpi=800)
import warnings
warnings.filterwarnings('ignore')

greatestPalette = ["#f44336","#265bcf","#36b3ba","#ffeb3b","#e91e63","#00cc92","#4caf50","#ffb65c","#9c27b0","#03a9f4","#43d61f","#ff9800","#673ab7","#cddc39","#81bdc5","#ff5722"]
palestPalette = ["#fcc9c5","#acb4e2","#2effea","#fffbd6","#f7abc5","#b1dafb","#b5deb6","#ffe79e","#d88ae5","#90dbfe","#d5e9be","#ffd699","#bca6e3","#70eeff","#edf3ba","#ffccbd"]
duckColors = ['#009F00','#003A99','#7FFF7F','#4DA0FF','#1A6DCC','#5800A4','#9FB8D5','#000766','#005200','#33EB33','#B3FFFF','#FFFF00','#80D3FF','#3B6FAB','#ED8D93','#B3B300']

#############################################################################
##         Flag to toggle rerunning the analysis or just plotting          ##
#############################################################################

rerun = True

#############################################################################
##                          Plot section flags                             ##
#############################################################################

do_ec_analysis = False
plot_raw = False
plot_normalized = False

redraw_featureplots = True
redraw_umaps = True

run_marker_analysis = True

expression_cutoff = 0.01  # Sets the threshold for min expression to be shown in plots

#############################################################################
## Change this to point toward your mount location for our MiStorage share ##
#############################################################################
mistorage_mount_point = '/home/mike/mistorage/'

#############################################################################
## Change this to list the sampleIDs of the samples you want to see        ##
#############################################################################
sample_list = ['150-1','150-2','150-3','150-4','150-5','150-6','150-7','150-8']


#############################################################################
## Change this to contain all genes you want to see expression for         ##
#############################################################################
genes_of_interest = ['EPCAM','VIM','CDX2','EGF','EGFR','ERBB2','ERBB3','ERBB4','NRG1','RSPO1','RSPO2','RSPO3','WNT2','WNT2B','WNT3','ACTA2','COL1A1','SOX17','T','TOP2A','WT1',
'CDH5','PECAM1','VWF','KDR','CD34','RGS5', 'CSPG4','ITGAM','PTPRC','HBB','STMN2',
'S100B','TUBB3','SPDEF','CHGA','LYZ','MUC2','MUC5','VIL1','ALPI','DPP4','LCT','SHH','IHH','DHH', 'HHIP',
'GLI1','GLI2','SMO','PTCH1','LGR5','OLFM4','PDGFRA','ADAMDEC1','ADAM17','ADAM10',
'ADAM7','DLL1','F3','NPY','GPX3','BMP4','ANO1','KIT','HAND2','NKX2-5','MSLN','WT1','UPK3B',
'HES1', 'HEY1','ID1','ID2','ID3','ID4','NOG','CHRD','GREM1','FOXF1','FOXF2','FOXL1',
'FOXL2','VEGFA','LOXL2','LAMC1','CYGB','FRZB','CTGF','CTSC','C1S','SYNPO2','EVA1B',
'ACKR3','XIST']

epi_cell_type_genes = ['EPCAM','CDX2','LGR5', 'OLFM4', 'CHGA', 'LYZ', 'MUC2', 'MUC5', 'VIL1', 'DPP4', 'FABP1', 'ALPI', 'SI', 'MGAM', 'SPDEF', 'SHH', 'IHH', 'PDGFA', 'SOX9','AXIN2','LGR5','ASCL2','CMYC','CD44','CYCLIND','TCF1','LEF1','PPARD','CJUN','MMP7','MSI1','LRP5','LRP6','DEFA5','DEFA4','REG3A','REG3B','REG3G']

#############################################################################
##        Adjust these parameters to get a nicer looking UMAP plot         ##
#############################################################################
# UMAP arguments
num_neighbors_use = 50
num_pcs_use = 15
umap_spread = 1
umap_min_dist = 0.3
maxiter = None
umap_gamma=1

# Louvain arguments
louv_res = 0.8

# PAGA arguments
size=20
paga_layout='fr'
threshold=0.005
node_size_scale=3

#############################################################################
##               DO NOT CHANGE ANYTHING BEYOND THIS POINT                  ##
#############################################################################




## Location to output the anndata h5ad files
raw_data_file = ''.join(['./data/Data_', '_'.join(sample_list), '.scanpy.raw.h5ad'])  # the file that will store the raw combined data
results_file = ''.join(['./data/Data_', '_'.join(sample_list), '.processed.h5ad'])  # the file that will store the analysis results
filtered_data_file = ''.join(['./data/Data_', '_'.join(sample_list), '.scanpy.filtered.h5ad'])  # the file that will store the raw combined data
endo_results_file = ''.join(['./data/Data_', '_'.join(sample_list), '.endothelial.processed.h5ad'])  # the file that will store the analysis results
neuro_results_file = ''.join(['./data/Data_', '_'.join(sample_list), '.neuronal.processed.h5ad'])  # the file that will store the analysis results

## Define function to generate a color gradient from a defined starting and ending color
def make_cmap(colors, position=None, bit=False):
	'''
	make_cmap takes a list of tuples which contain RGB values. The RGB
	values may either be in 8-bit [0 to 255] (in which bit must be set to
	True when called) or arithmetic [0 to 1] (default). make_cmap returns
	a cmap with equally spaced colors.
	Arrange your tuples so that the first color is the lowest value for the
	colorbar and the last is the highest.
	position contains values from 0 to 1 to dictate the location of each color.
	'''
	import matplotlib as mpl
	import numpy as np
	bit_rgb = np.linspace(0,1,256)
	if position == None:
		position = np.linspace(0,1,len(colors))
	else:
		if len(position) != len(colors):
			sys.exit("position length must be the same as colors")
		elif position[0] != 0 or position[-1] != 1:
			sys.exit("position must start with 0 and end with 1")
	if bit:
		for i in range(len(colors)):
			colors[i] = (bit_rgb[colors[i][0]],
						 bit_rgb[colors[i][1]],
						 bit_rgb[colors[i][2]])
	cdict = {'red':[], 'green':[], 'blue':[]}
	for pos, color in zip(position, colors):
		cdict['red'].append((pos, color[0], color[0]))
		cdict['green'].append((pos, color[1], color[1]))
		cdict['blue'].append((pos, color[2], color[2]))

	cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
	return cmap

annotation_dict = dict()

for line in open(''.join([mistorage_mount_point, '01_RNAseq_RAW_Data/single_cell_meta_data_table.tsv']), 'r'):
	#print(line)
	elem = str.split(line.rstrip())
	#print(elem)
	if elem:
		if elem[0] not in annotation_dict:
			annotation_dict[elem[0]] = elem[1:]

def Create_Scanpy_Anndata(mistorage_mount_point, sampleID):
	metadata_list = annotation_dict[sampleID][1:]
	newAdata = sc.read_10x_h5(''.join([mistorage_mount_point, annotation_dict[sampleID][0]]), genome='hg19')
	## Set gene names to be unique since there seem to be duplicate names from Cellranger
	newAdata.var_names_make_unique()
	## Add metadata for each sample to the observation (cells) annotations in the Anndata objects
	print('\nAdding Metadata for sample',sampleID,'\n')
	for field in metadata_list:
		field_list = str.split(field, ':')
		meta_name = field_list[0]
		meta_value = field_list[1]
		newAdata.obs[meta_name] = meta_value
	return(newAdata)

# function to get unique values 
def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return(unique_list)

## Create my custom palette for FeaturePlots and define a matlplotlib colormap object
#feature_colors = [(230,230,230), (35,35,142), (255,127,0)]
#position=[0, expression_cutoff, 1]
#my_feature_cmap = make_cmap(feature_colors, position=position, bit=True)
#dot_colors = [(230,230,230), (153,0,0), (255,145,0)]
#my_dot_cmap = make_cmap(dot_colors, position=position, bit=True)

#feature_colors = [(210,210,210), (210,210,210), (0,51,102), (255,141,41)]
#position=[0, 0.019999, 0.02, 1]
#my_feature_cmap = make_cmap(feature_colors, position=position, bit=True)
#dot_position=[0, 0.019999, 0.02, 1]
#dot_colors = [(210,210,210), (210,210,210), (0,51,102), (255,141,41)]
#my_dot_cmap = make_cmap(dot_colors, position=dot_position, bit=True)

feature_colors = [(210,210,210), (210,210,210), (245,245,200), (100,200,225), (0,45,125)]
position=[0, 0.019999, 0.02, 0.55, 1]
my_feature_cmap = make_cmap(feature_colors, position=position, bit=True)
dot_colors = [(210,210,210), (210,210,210), (245,245,200), (100,200,225), (0,45,125)]
my_dot_cmap = make_cmap(dot_colors, position=position, bit=True)




adata = sc.read_h5ad('/mnt/black/scRNA-seq/scanpy/Scanpy_NRG1_enteroids/data/Data_150-1_150-2_150-3_150-4_150-5_150-6_150-7_150-8.processed.h5ad')

adata0 = adata[adata.obs['sampleName'].isin(['HT323_EGF-0-NRG1-0'])]
adata1 = adata[adata.obs['sampleName'].isin(['HT323_EGF-0-NRG1-100'])]
adata2 = adata[adata.obs['sampleName'].isin(['HT323_EGF-1-NRG1-100'])]
adata3 = adata[adata.obs['sampleName'].isin(['HT323_EGF-10-NRG1-100'])]
adata4 = adata[adata.obs['sampleName'].isin(['HT323_EGF-100-NRG1-100'])]

first_gene_list = ['FABP2','RBP2','SMOC2','BEX1','MSI1','ASCL2','YBX1','OLFM4','LYZ','PRSS1','TFF1','TFF2']
gene_list = ['FABP1','FABP2','RBP2','SMOC2','BEX1','MSI1','ASCL2','YBX1','OLFM4','LYZ','PRSS1','TFF1','TFF2']

df = pd.DataFrame({'sample':'EGF-0-NRG1-100', 'expression':adata1.raw[:, 'FABP1'].X, 'gene':'FABP1'})

for gene in first_gene_list:
	df = pd.concat([df, pd.DataFrame({'sample':'EGF-0-NRG1-100', 'expression':adata1.raw[:, gene].X, 'gene':gene})])

for gene in gene_list:
	df = pd.concat([df, pd.DataFrame({'sample':'EGF-1-NRG1-100', 'expression':adata2.raw[:, gene].X, 'gene':gene})])

for gene in gene_list:
	df = pd.concat([df, pd.DataFrame({'sample':'EGF-10-NRG1-100', 'expression':adata3.raw[:, gene].X, 'gene':gene})])

for gene in gene_list:
	df = pd.concat([df, pd.DataFrame({'sample':'EGF-100-NRG1-100', 'expression':adata4.raw[:, gene].X, 'gene':gene})])

df.to_csv(path_or_buf='sorted_dataframe_geneExpression.csv', sep=',')


boxFig = plt.figure(dpi=80, figsize=(14,6))
bp = boxFig.add_subplot(1,1,1)

bp = sns.boxplot(x="gene", y="expression",
            hue="sample", data=df)

sns.despine(offset=10, trim=True)

boxFig.savefig('boxPlotFig.pdf')



boxenFig = plt.figure(dpi=80, figsize=(14,6))
bp = boxenFig.add_subplot(1,1,1)

bp = sns.boxenplot(x="gene", y="expression", hue="sample", data=df, palette='Blues')

sns.despine(offset=10, trim=True)

boxenFig.savefig('roboPlotFig.pdf')




violinFig = plt.figure(dpi=80, figsize=(14,6))
bp = violinFig.add_subplot(1,1,1)

bp = sns.violinplot(x="gene", y="expression", hue="sample", data=df)

sns.despine(offset=10, trim=True)

violinFig.savefig('violinPlotFig.pdf')


'''
swarmFig = plt.figure(dpi=80, figsize=(14,6))
bp = swarmFig.add_subplot(1,1,1)

bp = sns.stripplot(x="gene", y="expression", hue='sample', data=df, dodge=True, jitter=0.05)

sns.despine(offset=10, trim=True)

swarmFig.savefig('swarmPlotFig.pdf')
'''

'''
df = pd.DataFrame({'SLC5A1':adata0.raw[:, 'SLC5A1'].X, 'FABP2':adata0.raw[:, 'FABP2'].X, 'ACSL5':adata0.raw[:, 'ACSL5'].X, 'YBX1':adata0.raw[:, 'YBX1'].X, 'OLFM4':adata0.raw[:, 'OLFM4'].X, 'LYZ':adata0.raw[:, 'LYZ'].X, 'TFF1':adata0.raw[:, 'TFF1'].X, 'TFF2':adata0.raw[:, 'TFF2'].X})
print(df.median(axis = 0))
'''

print('\nEGF-0-NRG1-100\n----------------')
df = pd.DataFrame({'FABP1':adata1.raw[:, 'FABP1'].X, 'FABP2':adata1.raw[:, 'FABP2'].X, 'RBP2':adata1.raw[:, 'RBP2'].X, 'YBX1':adata1.raw[:, 'YBX1'].X, 'OLFM4':adata1.raw[:, 'OLFM4'].X, 'LYZ':adata1.raw[:, 'LYZ'].X, 'PRSS1':adata1.raw[:, 'PRSS1'].X, 'TFF1':adata1.raw[:, 'TFF1'].X, 'TFF2':adata1.raw[:, 'TFF2'].X})
print(df.mean(axis = 0))
print(df.sem(axis = 0))

print('\nEGF-1-NRG1-100\n----------------')
df = pd.DataFrame({'FABP1':adata2.raw[:, 'FABP1'].X, 'FABP2':adata2.raw[:, 'FABP2'].X, 'RBP2':adata2.raw[:, 'RBP2'].X, 'YBX1':adata2.raw[:, 'YBX1'].X, 'OLFM4':adata2.raw[:, 'OLFM4'].X, 'LYZ':adata2.raw[:, 'LYZ'].X, 'PRSS1':adata2.raw[:, 'PRSS1'].X, 'TFF1':adata2.raw[:, 'TFF1'].X, 'TFF2':adata2.raw[:, 'TFF2'].X})
print(df.mean(axis = 0))
print(df.sem(axis = 0))

print('\nEGF-10-NRG1-100\n----------------')
df = pd.DataFrame({'FABP1':adata3.raw[:, 'FABP1'].X, 'FABP2':adata3.raw[:, 'FABP2'].X, 'RBP2':adata3.raw[:, 'RBP2'].X, 'YBX1':adata3.raw[:, 'YBX1'].X, 'OLFM4':adata3.raw[:, 'OLFM4'].X, 'LYZ':adata3.raw[:, 'LYZ'].X, 'PRSS1':adata3.raw[:, 'PRSS1'].X, 'TFF1':adata3.raw[:, 'TFF1'].X, 'TFF2':adata3.raw[:, 'TFF2'].X})
print(df.mean(axis = 0))
print(df.sem(axis = 0))

print('\nEGF-100-NRG1-100\n----------------')
df = pd.DataFrame({'FABP1':adata4.raw[:, 'FABP1'].X, 'FABP2':adata4.raw[:, 'FABP2'].X, 'RBP2':adata4.raw[:, 'RBP2'].X, 'YBX1':adata4.raw[:, 'YBX1'].X, 'OLFM4':adata4.raw[:, 'OLFM4'].X, 'LYZ':adata4.raw[:, 'LYZ'].X, 'PRSS1':adata4.raw[:, 'PRSS1'].X, 'TFF1':adata4.raw[:, 'TFF1'].X, 'TFF2':adata4.raw[:, 'TFF2'].X})
print(df.mean(axis = 0))
print(df.sem(axis = 0))

df.to_csv(path_or_buf='dataframe_geneExpression.csv', sep=',')

'''
df = pd.DataFrame({'sample':'EGF-0-NRG1-100', 'FABP1':adata1.raw[:, 'FABP1'].X, 'FABP2':adata1.raw[:, 'FABP2'].X, 'RBP2':adata1.raw[:, 'RBP2'].X, 'YBX1':adata1.raw[:, 'YBX1'].X, 'OLFM4':adata1.raw[:, 'OLFM4'].X, 'LYZ':adata1.raw[:, 'LYZ'].X, 'PRSS1':adata1.raw[:, 'PRSS1'].X, 'TFF1':adata1.raw[:, 'TFF1'].X, 'TFF2':adata1.raw[:, 'TFF2'].X})

df = pd.concat([df, pd.DataFrame({'sample':'EGF-1-NRG1-100', 'FABP1':adata2.raw[:, 'FABP1'].X, 'FABP2':adata2.raw[:, 'FABP2'].X, 'RBP2':adata2.raw[:, 'RBP2'].X, 'YBX1':adata2.raw[:, 'YBX1'].X, 'OLFM4':adata2.raw[:, 'OLFM4'].X, 'LYZ':adata2.raw[:, 'LYZ'].X, 'PRSS1':adata2.raw[:, 'PRSS1'].X, 'TFF1':adata2.raw[:, 'TFF1'].X, 'TFF2':adata2.raw[:, 'TFF2'].X})], axis=0)

df = pd.concat([df, pd.DataFrame({'sample':'EGF-10-NRG1-100', 'FABP1':adata3.raw[:, 'FABP1'].X, 'FABP2':adata3.raw[:, 'FABP2'].X, 'RBP2':adata3.raw[:, 'RBP2'].X, 'YBX1':adata3.raw[:, 'YBX1'].X, 'OLFM4':adata3.raw[:, 'OLFM4'].X, 'LYZ':adata3.raw[:, 'LYZ'].X, 'PRSS1':adata3.raw[:, 'PRSS1'].X, 'TFF1':adata3.raw[:, 'TFF1'].X, 'TFF2':adata3.raw[:, 'TFF2'].X})], axis=0)

df = pd.concat([df, pd.DataFrame({'sample':'EGF-100-NRG1-0', 'FABP1':adata4.raw[:, 'FABP1'].X, 'FABP2':adata4.raw[:, 'FABP2'].X, 'RBP2':adata4.raw[:, 'RBP2'].X, 'YBX1':adata4.raw[:, 'YBX1'].X, 'OLFM4':adata4.raw[:, 'OLFM4'].X, 'LYZ':adata4.raw[:, 'LYZ'].X, 'PRSS1':adata4.raw[:, 'PRSS1'].X, 'TFF1':adata4.raw[:, 'TFF1'].X, 'TFF2':adata4.raw[:, 'TFF2'].X})], axis=0)
'''

df.to_csv(path_or_buf='appended_dataframe_geneExpression.csv', sep=',')













