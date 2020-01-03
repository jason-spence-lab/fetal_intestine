import numpy as np
import pandas as pd
import bbknn
import scanpy as sc
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
sc.settings.verbosity = 3			 # verbosity: errors (0), warnings (1), info (2), hints (3)
import warnings
warnings.filterwarnings('ignore')
sc.settings.set_figure_params(dpi_save=300, dpi=300)

greatestPalette = ["#f44336","#265bcf","#36b3ba","#ffeb3b","#e91e63","#00cc92","#4caf50","#ffb65c","#9c27b0","#03a9f4","#43d61f","#ff9800","#673ab7","#cddc39","#81bdc5","#ff5722"]
palestPalette = ["#fcc9c5","#acb4e2","#2effea","#fffbd6","#f7abc5","#b1dafb","#b5deb6","#ffe79e","#d88ae5","#90dbfe","#d5e9be","#ffd699","#bca6e3","#70eeff","#edf3ba","#ffccbd"]

fig3_palette = ['#4476A5','#2BB673']

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
sample_list = ['3011-1','3011-3']


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

epi_cell_type_genes = ['EPCAM','CDX2','LGR5', 'OLFM4', 'CHGA', 'LYZ', 'MUC2', 'MUC5', 'VIL1', 'DPP4', 'FABP1', 'ALPI', 'SI', 'MGAM', 'SPDEF', 'SHH', 'IHH', 'PDGFA', 'SOX9','AXIN2','LGR5','ASCL2','CMYC','CD44','CYCLIND','TCF1','LEF1','PPARD','CJUN','MMP7','MSI1','LRP5','LRP6']

erbb_pathway = ['EGF','EGFR','ERBB2','ERBB3','ERBB4','TGFA','HBEGF','AREG','BTC','EPGN','EREG','NRG1','NRG2','NRG3','NRG4']

#############################################################################
##        Adjust these parameters to get a nicer looking UMAP plot         ##
#############################################################################
# UMAP arguments
num_neighbors_use = 25
num_pcs_use = 10
umap_spread = 1
umap_min_dist = 0.3
maxiter = None
umap_gamma=1

# Louvain arguments
louv_res = 0.6

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




## Read the raw Cellranger filtered data matrices into new Anndata objects
def runBasicAnalysis():
	
	first_adata = False
	
	if Path(raw_data_file).is_file():
		print(''.join(['Data_', '_'.join(sample_list), '.scanpy.raw.h5ad']), 'found, using this existing raw data file\n')
		adata = sc.read_h5ad(raw_data_file)
	else:
		print('\nNo existing h5ad raw data file found, reading in 10x h5 data for each sample\n')
		for sample in sample_list:
			if not first_adata:
				adata = Create_Scanpy_Anndata(mistorage_mount_point, sample)
				first_adata = True
			else:
				adata = adata.concatenate(Create_Scanpy_Anndata(mistorage_mount_point, sample))
		
		## Make cell names unique by adding _1, _2, _3 sequentially to each duplicated 10x barcode/name
		adata.obs_names_make_unique()
		## Write the raw combined dataset to disk so you can skip combining samples next time
		print('\nSaving raw combined sample data to', raw_data_file, '\n')
		adata.write(raw_data_file)
		open('./data/Prefiltered_gene_list.txt', 'w').write('\n'.join(adata.var_names.tolist()))
	
	## Basic filtering to get rid of useless cells and unexpressed genes
	
	sc.pp.filter_cells(adata, min_genes=1000)
	sc.pp.filter_genes(adata, min_cells=5)
	
	print('\nDoing initial filtering...\nKeeping', len(adata.obs_names), 'cells and', len(adata.var_names), 'genes.\n')
	
	mito_genes = adata.var_names.str.startswith('MT-')
	# Calculate the percent of genes derived from mito vs genome
	# the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
	adata.obs['percent_mito'] = np.sum(
		adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
	# add the total counts per cell as observations-annotation to adata
	adata.obs['n_counts'] = adata.X.sum(axis=1).A1
	
	sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
				 jitter=0.4, multi_panel=True, save = '_preFiltering_plot.pdf', show = False)
	
	## Actually do the filtering.
	
	adata = adata[adata.obs['n_genes'] > 1000, :]   # Keep cells with more than 1000 genes
	adata = adata[adata.obs['n_genes'] < 9500, :]   # Keep cells with less than 5000 genes to remove most doublets
	adata = adata[adata.obs['n_counts'] < 125000, :] # Keep cells with less than 15000 UMIs to catch a few remaining doublets
	adata = adata[adata.obs['percent_mito'] < 0.8, :]   # Keep cells with less than 0.1 mito/genomic gene ratio
	sc.pp.filter_genes(adata, min_cells=5)	# Refilter genes to get rid of genes that are only in a tiny number of cells
	
	print('\nDoing final filtering...\nKeeping', len(adata.obs_names), 'cells and', len(adata.var_names), 'genes.\n')
	
	open('./data/Final_filtered_gene_list.txt', 'w').write('\n'.join(adata.var_names.tolist()))
	
	sc.pl.violin(adata, ['n_genes','n_counts', 'percent_mito'],
				 jitter=0.4, multi_panel=True, save = '_postFiltering_plot.pdf', show = False)
	
	## Normalize the expression matrix to 10,000 reads per cell, so that counts become comparable among cells.
	# This corrects for differences in sequencing depth between cells and samples
	
	#sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.normalize_total(adata)
	
	## Log transform the data.
	
	sc.pp.log1p(adata)
	
	## Set the .raw attribute of AnnData object to the logarithmized raw gene expression for later use in differential testing and visualizations of gene expression.
	# We need to do this because the expression matrix will be rescaled and centered which flattens expression too much for some purposes
	
	adata.write(filtered_data_file)
	adata.raw = adata
	
	## Identify highly-variable genes based on dispersion relative to expression level.
	
	sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=6, min_disp=0.2)
	
	## Filter the genes to remove non-variable genes since they are uninformative
	
	adata = adata[:, adata.var['highly_variable']]
	
	## Regress out effects of total reads per cell and the percentage of mitochondrial genes expressed.
	
	sc.pp.regress_out(adata, ['n_counts'])
	
	## Scale each gene to unit variance. Clip values exceeding standard deviation 10 to remove extreme outliers
	
	sc.pp.scale(adata, max_value=10)
	
	## Run PCA to compute the default number of components
	
	sc.tl.pca(adata, svd_solver='arpack')
	
	## Rank genes according to contributions to PCs.
	
	sc.pl.pca_loadings(adata, show=False, components=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], save='_PCA-loadings.pdf')
	
	## Draw the PCA elbow plot to determine which PCs to use
	sc.pl.pca_variance_ratio(adata, log=True, n_pcs = 100, save = '_elbowPlot.pdf', show = False)
	
	## Compute nearest-neighbors
	
	sc.pp.neighbors(adata, n_neighbors=num_neighbors_use, n_pcs=num_pcs_use)
	
	#bbknn.bbknn(adata, batch_key='sampleName', n_pcs=25, neighbors_within_batch=3, copy=False)
	
	## Calculate cell clusters via Louvain algorithm
	
	sc.tl.louvain(adata, resolution = louv_res)
	
	## Run UMAP Dim reduction
	
	sc.tl.umap(adata, min_dist=umap_min_dist, maxiter=maxiter, spread=umap_spread, gamma=umap_gamma)
	
	## Save the final clustered and dim-reduced data file as h5ad
	print('\nSaving processed Anndata data to', results_file, '\n')
	adata.write(results_file)
	return(adata)


if rerun:
	adata = runBasicAnalysis()
	adata.raw = sc.read_h5ad(filtered_data_file)
	print('Rerunning filtering and normalization...')
elif not rerun:
	print('Rerun toggled to "off"...\nMoving directly to subclustering and plotting\n') 
	adata = sc.read_h5ad(results_file)
	adata.raw = sc.read_h5ad(filtered_data_file)

orig_adata = sc.read_h5ad(raw_data_file)








if redraw_umaps:
	print('\nRedrawing the umap plots...\n---------------------------\n')
	sc.pl.umap(adata, color='louvain', save = '_clusterIdentity.pdf', show = False, legend_loc = 'on data', edges = True, edges_color = 'lightgrey', edges_width = 0.01, size = 20, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	sc.pl.umap(adata, color='louvain', save = '_clusterIdentity_noEdge.pdf', show = False, legend_loc = 'on data', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = 20, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	sc.pl.umap(adata, color=['louvain', 'media'], save = '_clusterIdentity_media.pdf', show = False, legend_loc = 'right margin', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = 20, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	sc.pl.umap(adata, color='age', save = '_age.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='tissue', save = '_tissue.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='sex', save = '_sex.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='gel', save = '_hydrogel.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='media', save = '_media.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = fig3_palette, alpha = 0.95)
	sc.pl.umap(adata, color='sampleName', save = '_sample.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = fig3_palette, alpha = 0.95)
	
	## Run PAGA to get nicer cell starting positions for UMAP
	sc.tl.paga(adata, groups='louvain')
	sc.pl.paga(adata, color='louvain', save='_paga-init-pattern.pdf', show=False, threshold=threshold, node_size_scale=node_size_scale, node_size_power=0.9, layout=paga_layout)
	
	adata.write(results_file)







if redraw_featureplots:
	print('\nRedrawing the feature plots...\n---------------------------\n')
	## Do FeaturePlots for select genes	
	expressed_dict = dict()
	
	for gene in adata.raw.var_names.values.tolist():
		if gene not in expressed_dict:
			expressed_dict[str(gene)] = 1
			#print(gene)
	
	genes_to_plot = []
	paneth_genes = ['DEFA5','DEFA6','LYZ']
	
	for gene in paneth_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting standard marker genes:', ' '.join(genes_to_plot),'\n')
	
	sc.pl.umap(adata, color=genes_to_plot, save = '_paneth-lysozyme_featureplots.png', show = False, cmap = my_feature_cmap, size = 25, use_raw = True)
	
	
	genes_to_plot = []
	
	for gene in genes_of_interest:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ' '.join(genes_to_plot),'\n')
	
	sc.pl.umap(adata, color=genes_to_plot, save = '_featureplots.png', show = False, cmap = my_feature_cmap, size = 25, use_raw = True)
	
	genes_to_plot = []
	standard_marker_genes = ['EPCAM', 'VIM', 'ACTA2', 'TOP2A', 'WT1', 'CDH5', 'DCN', 'KDR', 'ITGAM', 'PTPRC', 'HBB', 'STMN2', 'S100B']
	
	for gene in standard_marker_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting standard marker genes:', ' '.join(genes_to_plot),'\n')
	
	sc.pl.umap(adata, color=genes_to_plot, save = '_standard_markers_featureplots.png', show = False, cmap = my_feature_cmap, size = 25, use_raw = True)
	
	genes_to_plot = []
	
	for gene in epi_cell_type_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting standard marker genes:', ' '.join(genes_to_plot),'\n')
	
	sc.pl.umap(adata, color=genes_to_plot, save = '_epi_cell_type_genes_featureplots.png', show = False, cmap = my_feature_cmap, size = 25, use_raw = True)
	
	genes_to_plot = []
	
	erbb_pathway = ['EGF','EGFR','ERBB2','ERBB3','ERBB4','TGFA','HBEGF','AREG','BTC','EPGN','EREG','NRG1','NRG2','NRG3','NRG4']
	
	for gene in erbb_pathway:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting ERBB pathway genes:', ' '.join(genes_to_plot),'\n')
	
	sc.pl.umap(adata, color=genes_to_plot, save = '_ERBB_pathway_genes_featureplots.png', show = False, cmap = my_feature_cmap, size = 25, use_raw = True)

print('Starting gene expression plotting...\n------------------------------------\n')


print('Checking for expression of genes of interest\n')
expressed_dict = dict()

for gene in adata.raw.var_names.values.tolist():
	if gene not in expressed_dict:
		expressed_dict[str(gene)] = 1

genes_to_plot = []
	
for gene in genes_of_interest:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Found cells expressing', ' '.join(genes_to_plot), '\n')


if run_marker_analysis:
	print("\nAll done with general workflow... now finding marker genes.\n")
	## Find marker genes via Wilxocon test based on Louvain cluster assignment
	# Create a simple plot to show the top 25 most significant markers for each cluster
	
	sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon')
	
	sc.tl.filter_rank_genes_groups(adata, groupby='louvain', use_raw=True, log=True, key_added='rank_genes_groups_filtered', min_in_group_fraction=0.5, min_fold_change=2, max_out_group_fraction=0.5)
	
	sc.pl.rank_genes_groups(adata, key='rank_genes_groups_filtered', n_genes=30, sharey=False, save = '_markerPlots.pdf', show = False)
	sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups_filtered',  n_genes=10, save = '_markerDotPlots.pdf', color_map=my_dot_cmap, show = False, mean_only_expressed=True, dot_min=0.2, dot_max=1, standard_scale='var')



fig_4B1_genes = ['LGR5','OLFM4']
	
genes_to_plot = []

for gene in fig_4B1_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='media', use_raw=True, mean_only_expressed=True, save = '_fig4B1_DotPlot_logScale.pdf', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=False, log=True)

fig_4B2_genes = ['MKI67','TOP2A']
	
genes_to_plot = []

for gene in fig_4B2_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='media', use_raw=True, mean_only_expressed=True, save = '_fig4B2_DotPlot_logScale.pdf', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=False, log=True)



fig_4B_umap_genes = ['MKI67','TOP2A','LGR5','OLFM4']
	
genes_to_plot = []

for gene in fig_4B_umap_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.umap(adata, color=genes_to_plot, save = '_fig3B_UMAP_genes_featureplots.png', show = False, cmap = my_feature_cmap, size = 25, use_raw = True)

fig_3C_umap_genes = ['OLFM4', 'MKI67']
	
genes_to_plot = []

for gene in fig_3C_umap_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.umap(adata, color=genes_to_plot, save = '_fig3C_UMAP_genes_featureplots.png', show = False, cmap = my_feature_cmap, size = 25, use_raw = True)


sc.tl.rank_genes_groups(adata, 'sampleName', method='wilcoxon')

sc.tl.filter_rank_genes_groups(adata, groupby='sampleName', use_raw=True, log=True, key_added='rank_genes_groups_filtered', min_in_group_fraction=0.1, min_fold_change=1, max_out_group_fraction=0.90)
sc.pl.rank_genes_groups(adata, key='rank_genes_groups_filtered', n_genes=20, sharey=False, save = '_markerPlots.pdf', show = False)
sc.pl.rank_genes_groups_heatmap(adata, key='rank_genes_groups_filtered', groupby='sampleName',  n_genes=20, save = '_growthFactor_markerHeatmap.pdf', show = False)







print('\nDone with entire script execution')










