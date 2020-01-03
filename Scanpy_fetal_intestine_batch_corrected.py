import sys
import bbknn
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from pathlib import Path
import mjc_functions as mjc
import matplotlib.pyplot as plt
import scanorama
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 3			 # verbosity: errors (0), warnings (1), info (2), hints (3)
sns.set(style="white", color_codes=True)
sc.settings.set_figure_params(dpi_save=300, dpi=300, fontsize=8)


greatestPalette = ["#f44336","#265bcf","#36b3ba","#ffeb3b","#e91e63","#00cc92",
"#4caf50","#ffb65c","#9c27b0","#03a9f4","#43d61f","#ff9800","#673ab7","#cddc39",
"#81bdc5","#ff5722","#fcc9c5","#acb4e2","#2effea","#fffbd6","#f7abc5","#b1dafb",
"#b5deb6","#ffe79e","#d88ae5","#90dbfe","#d5e9be","#ffd699","#bca6e3","#70eeff",
"#edf3ba","#ffccbd"]

palestPalette = ["#fcc9c5","#acb4e2","#2effea","#fffbd6","#f7abc5","#b1dafb","#b5deb6","#ffe79e","#d88ae5","#90dbfe","#d5e9be","#ffd699","#bca6e3","#70eeff","#edf3ba","#ffccbd"]



#############################################################################
##         Flag to toggle rerunning the analysis or just plotting          ##
#############################################################################

rerun = True

#############################################################################
##                          Plot section flags                             ##
#############################################################################

redraw_featureplots = True
redraw_umaps = True

run_marker_analysis = True

expression_cutoff = 0.01  # Sets the threshold for min expression to be shown in plots

#############################################################################
## Change this to point toward your mount location for our MiStorage share ##
#############################################################################
mistorage_mount_point = '/home/mike/mistorage/'

#############################################################################
##        Adjust these parameters to get a nicer looking UMAP plot         ##
#############################################################################
# UMAP arguments
num_neighbors_use = 20
num_pcs_use = 9
umap_spread = 1
umap_min_dist = 0.5
maxiter = None
umap_gamma=1
random_state = 20120608

paga_init = True

dot_size = 10 # size of each point un UMAP scatters

# Louvain arguments
louv_res = 0.8

# PAGA arguments
size=20
paga_layout='fr'
threshold=0.01
node_size_scale=3

#############################################################################
## Change this to contain all genes you want to see expression for         ##
#############################################################################
genes_of_interest = ['EPCAM','VIM','CDX2','ACTA2','COL1A1','SOX17','T','TOP2A','WT1','CDH5','PECAM1','VWF','KDR','CD34','RGS5', 
'CSPG4','ITGAM','PTPRC','HBB','STMN2','S100B','TUBB3','SPDEF','CHGA','LYZ','MUC2','MUC5','VIL1','SHH','IHH','DHH', 
'HHIP','GLI1','GLI2','SMO','PTCH1','LGR5','OLFM4','PDGFRA','DLL1','F3','NPY','GPX3','BMP4','ANO1','KIT','HAND2','WT1','UPK3B','HES1', 
'HEY1','ID1','ID2','ID3','ID4','NOG','CHRD','GREM1','FOXF1','FOXF2','FOXL1','FOXL2','VEGFA','LOXL2','LAMC1','CYGB','FRZB',
'CTGF','CTSC','C1S','SYNPO2','EVA1B','ACKR3','XIST','DKK1','EGF','NRG1','EGFR']

mes_score_genes = ['TCF21','COL1A1','COL1A2','DCN','ACTA2','TAGLN']
subMucosa_score_genes = ['DLL1','CXCL14','F3','BMP4']
epi_score_genes = ['EPCAM','CDH1','CDX2']
immune_score_genes = ['PTPRC','HBB','CORO1A','ARHGDIB']
endo_score_genes = ['KDR','CDH5','VWF','CLDN5']
nuro_score_genes = ['S100B','ELAVL4','TUBB2B']
subEpi_score_genes = ['F3','NPY','NRG1']

#############################################################################
##                          End of global settings                         ##
#############################################################################

## Create my custom palette for FeaturePlots and define a matlplotlib colormap object
#feature_colors = [(230,230,230), (35,35,142), (255,127,0)]
feature_colors = [(210,210,210), (210,210,210), (245,245,200), (100,200,225), (0,45,125)]
position=[0, 0.019999, 0.02, 0.55, 1]
my_feature_cmap = mjc.make_cmap(feature_colors, position=position, bit=True)
dot_colors = [(210,210,210), (210,210,210), (245,245,200), (100,200,225), (0,45,125)]
my_dot_cmap = mjc.make_cmap(dot_colors, position=position, bit=True)

position=[0, 0.019999, 0.4, 0.55, 1]
dot_colors = [(210,210,210), (210,210,210), (210,210,210), (100,200,225), (0,45,125)]
ct_score_cmap = mjc.make_cmap(dot_colors, position=position, bit=True)




if Path('./data/Raw.concatenated.anndata.h5ad').is_file():
	print('Found [./data/Raw.concatenated.anndata.h5ad] loading data from there')
	adata = sc.read_h5ad('./data/Raw.concatenated.anndata.h5ad')
	print('\nConcatenated samples contain...\n', len(adata.obs_names), 'cells and', len(adata.var_names), 'genes.\n')
	
else:
	mapping_genome = 'hg19'
	
	adata_47 = mjc.Create_Scanpy_Anndata(mistorage_mount_point, '2598-31', mapping_genome)
	#sc.pp.downsample_counts(adata_47, counts_per_cell=5000, random_state=20120608, replace=False, copy=False)
	adata_47 = mjc.Filter_New_Anndata(adata_47, 'Day_47')
	
	adata_59 = mjc.Create_Scanpy_Anndata(mistorage_mount_point, '2757-2', mapping_genome)
	#sc.pp.downsample_counts(adata_59, counts_per_cell=5000, random_state=20120608, replace=False, copy=False)
	adata_59 = mjc.Filter_New_Anndata(adata_59, 'Day_59')
	
	adata_72 = mjc.Create_Scanpy_Anndata(mistorage_mount_point, '2856-1', mapping_genome)
	#sc.pp.downsample_counts(adata_72, counts_per_cell=5000, random_state=20120608, replace=False, copy=False)
	adata_72 = mjc.Filter_New_Anndata(adata_72, 'Day_72')
	
	adata_80 = mjc.Create_Scanpy_Anndata(mistorage_mount_point, '2598-24', mapping_genome)
	#sc.pp.downsample_counts(adata_80, counts_per_cell=5000, random_state=20120608, replace=False, copy=False)
	adata_80 = mjc.Filter_New_Anndata(adata_80, 'Day_80')
	
	adata_101 = mjc.Create_Scanpy_Anndata(mistorage_mount_point, '2511-2', mapping_genome)
	#sc.pp.downsample_counts(adata_101, counts_per_cell=5000, random_state=20120608, replace=False, copy=False)
	adata_101 = mjc.Filter_New_Anndata(adata_101, 'Day_101')
	
	adata_122 = mjc.Create_Scanpy_Anndata(mistorage_mount_point, '2182-5', mapping_genome).concatenate(mjc.Create_Scanpy_Anndata(mistorage_mount_point, '2182-6', mapping_genome))
	#sc.pp.downsample_counts(adata_122, counts_per_cell=5000, random_state=20120608, replace=False, copy=False)
	adata_122 = mjc.Filter_New_Anndata(adata_122, 'Day_122')
	
	adata_127 = mjc.Create_Scanpy_Anndata(mistorage_mount_point, '2250-1', mapping_genome).concatenate(mjc.Create_Scanpy_Anndata(mistorage_mount_point, '2250-2', mapping_genome))
	#sc.pp.downsample_counts(adata_127, counts_per_cell=5000, random_state=20120608, replace=False, copy=False)
	adata_127 = mjc.Filter_New_Anndata(adata_127, 'Day_127')
	
	adata_132 = mjc.Create_Scanpy_Anndata(mistorage_mount_point, '2598-28', mapping_genome)
	#sc.pp.downsample_counts(adata_132, counts_per_cell=5000, random_state=20120608, replace=False, copy=False)
	adata_132 = mjc.Filter_New_Anndata(adata_132, 'Day_132')
	
	adata = adata_47.concatenate(adata_59, adata_72, adata_80, adata_101, adata_122, adata_127, adata_132)
	
	adata.write('./data/Raw.concatenated.anndata.h5ad')
	
	print('\nConcatenated samples contain...\n', len(adata.obs_names), 'cells and', len(adata.var_names), 'genes.\n')


#sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.normalize_total(adata)

## Log transform the data.
sc.pp.log1p(adata)

## Set the .raw attribute of AnnData object to the logarithmized raw gene expression for later use in differential testing and visualizations of gene expression.
# We need to do this because the expression matrix will be rescaled and centered which flattens expression too much for some purposes
adata.write('./data/Filtered.concatenated.anndata.h5ad')
adata.raw = adata

sc.tl.score_genes(adata, mes_score_genes, ctrl_size=25, gene_pool=None, n_bins=25, score_name='mes_score', random_state=0, copy=False, use_raw=False)
sc.tl.score_genes(adata, subMucosa_score_genes, ctrl_size=25, gene_pool=None, n_bins=25, score_name='subMucosa_score', random_state=0, copy=False, use_raw=False)
sc.tl.score_genes(adata, epi_score_genes, ctrl_size=25, gene_pool=None, n_bins=25, score_name='epi_score', random_state=0, copy=False, use_raw=False)
sc.tl.score_genes(adata, immune_score_genes, ctrl_size=25, gene_pool=None, n_bins=25, score_name='immune_score', random_state=0, copy=False, use_raw=False)
sc.tl.score_genes(adata, endo_score_genes, ctrl_size=25, gene_pool=None, n_bins=25, score_name='endo_score', random_state=0, copy=False, use_raw=False)
sc.tl.score_genes(adata, nuro_score_genes, ctrl_size=25, gene_pool=None, n_bins=25, score_name='neuro_score', random_state=0, copy=False, use_raw=False)
sc.tl.score_genes(adata, subEpi_score_genes, ctrl_size=25, gene_pool=None, n_bins=25, score_name='subEpi_score', random_state=0, copy=False, use_raw=False)

## Identify highly-variable genes based on dispersion relative to expression level.
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=6, min_disp=0.2)

## Filter the genes to remove non-variable genes since they are uninformative
adata = adata[:, adata.var['highly_variable']]

## Regress out effects of total reads per cell and the percentage of mitochondrial genes expressed.
sc.pp.regress_out(adata, ['n_counts', 'S_score', 'G2M_score'])

## Scale each gene to unit variance. Clip values exceeding standard deviation 10 to remove extreme outliers
sc.pp.scale(adata, max_value=10)

## Run PCA to compute the default number of components
sc.tl.pca(adata, svd_solver='arpack')

## Rank genes according to contributions to PCs.
sc.pl.pca_loadings(adata, show=False, components=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], save='_PCA-loadings.png')

## Draw the PCA elbow plot to determine which PCs to use
sc.pl.pca_variance_ratio(adata, log=True, n_pcs = 100, save = '_elbowPlot.png', show = False)

## Compute nearest-neighbors
sc.pp.neighbors(adata, n_neighbors=num_neighbors_use, n_pcs=num_pcs_use)

## fix batch differences based on XX/XY
bbknn.bbknn(adata, batch_key='sampleName', n_pcs=75, neighbors_within_batch=3, copy=False)

## Calculate cell clusters via Louvain algorithm
sc.tl.louvain(adata, resolution = louv_res)

sc.tl.paga(adata, groups='louvain')
sc.pl.paga(adata, color='louvain', save=False, show=False, threshold=threshold, node_size_scale=node_size_scale, node_size_power=0.9, layout=paga_layout)

sc.tl.umap(adata, init_pos='paga', min_dist=umap_min_dist, maxiter=maxiter, spread=umap_spread, gamma=umap_gamma, random_state=random_state)
#sc.tl.umap(adata, min_dist=umap_min_dist, maxiter=maxiter, spread=umap_spread, gamma=umap_gamma, random_state=random_state)

#sc.pl.umap(adata, color='louvain', save = '_clusterIdentity.png', show = False, legend_loc = 'on data', edges = True, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
sc.pl.umap(adata, color='louvain', save = '_clusterIdentity_noEdge.png', show = False, legend_loc = 'on data', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
sc.pl.umap(adata, color=['louvain', 'age'], save = '_clusterIdentity_age.png', show = False, legend_loc = 'right margin', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
sc.pl.umap(adata, color='age', save = '_age.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95)
sc.pl.umap(adata, color='sex', save = '_sex.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95)
sc.pl.umap(adata, color='sampleName', save = '_sample.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95)
sc.pl.umap(adata, color=['n_genes','n_counts','percent_mito'], save = '_stats.png', show = False, edges = False, cmap = my_feature_cmap, size = dot_size+10)
sc.pl.umap(adata, color=['mes_score','subMucosa_score','epi_score','endo_score','immune_score','neuro_score','subEpi_score'], save = '_cellType_score.png', show = False, edges = False, cmap = ct_score_cmap, size = dot_size+10)
sc.pl.umap(adata, color='subEpi_score', save = '_subEpi_score.png', show = False, edges = False, cmap = ct_score_cmap, size = dot_size+10)



ageFigumap = plt.figure(dpi=80, figsize=(18,7))
ax1 = ageFigumap.add_subplot(1,3,1)
ax2 = ageFigumap.add_subplot(1,3,2)

sc.pl.umap(adata, color='louvain', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6, ax=ax1)
sc.pl.umap(adata, color='age', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6, ax=ax2)

ax1.set_title('Louvain clusters')
ax2.set_title('Age (days)')


ageFigumap.savefig('UMAP_louvain_age_panels.png')













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

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.umap(adata, color=genes_to_plot, save = '_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)




emilys_list1 = ['PGK1','CDH1','EPCAM','CDX2','VIL1','SLC2A5','ALPI','MUC2','MUC5AC','MUC5B','TFF3','LYZ','DEFA5','DEFA6','LGR5',
'TAGLN','COL1A1','COL2A1','VCL','VIM','ACTA2','PTPRC','CD163','CD80','CD86','ARG1','CD14','MRC1','MS4A1','CCR5','CD4','CD8A',
'CD8B','CR2','TGFB1','IGF1','IL6','IL1B','IGF1R','FGF2','HIF1A','TNF','TGFBR2','AXL','GAS6','MERTK','CD44','KLF1','TLR4','MAPK14']

emilys_list2 = ['MAPK1','MAPK3','RAF1','KRAS','AKT1','AKT2','AKT3','RHOA','ROCK1','ROCK2','SMAD1','SMAD2','SMAD3','SMAD4','SMAD5','SMAD6','SMAD7',
'SMAD9','TYK2','STAT3','STAT1','JAK2','HIF1A','MMP9','MTOR','SOCS1','SOCS3','SNAI1','TWIST1']

genes_to_plot = []
for gene in emilys_list1:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')
sc.pl.umap(adata, color=genes_to_plot, save = '_emilysGenes1_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)

genes_to_plot = []
for gene in emilys_list2:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')
sc.pl.umap(adata, color=genes_to_plot, save = '_emilysGenes2_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)





genes_to_plot = []
	
erbb_pathway = ['EGF','EGFR','ERBB2','ERBB3','ERBB4','TGFA','HBEGF','AREG','BTC','EPGN','EREG','NRG1','NRG2','NRG3','NRG4']

for gene in erbb_pathway:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting ERBB pathway genes:', ' '.join(genes_to_plot),'\n')

sc.pl.umap(adata, color=genes_to_plot, save = '_ERBB_pathway_genes_featureplots.png', show = False, cmap = my_feature_cmap, size = 25, use_raw = True)



genes_to_plot = []
	
erbb_pathway = ['COL1A1','ACTA2','TAGLN','DCN']

for gene in erbb_pathway:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting ERBB pathway genes:', ' '.join(genes_to_plot),'\n')

sc.pl.umap(adata, color=genes_to_plot, save = '_fig1_genes_featureplots.png', show = False, cmap = my_feature_cmap, size = 25, use_raw = True)


fig_2A_genes = ['ACTA2','TAGLN','DLL1','F3','NPY','GPX3']
	
genes_to_plot = []

for gene in fig_2A_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='age', mean_only_expressed=True, save = '_fig2A_DotPlot.png', show = False, color_map=my_dot_cmap, dendrogram=False)


fig_3B_umap_genes = ['EPCAM','ALPI','EGF','LGR5']
	
genes_to_plot = []

for gene in fig_3B_umap_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.umap(adata, color=genes_to_plot, save = '_fig3B_UMAP_genes_featureplots.png', show = False, cmap = my_feature_cmap, size = 25, use_raw = True)

fig_3B_genes = ['LGR5','OLFM4','FABP2','SI','DPP4','F3','NPY','ACTA2','TAGLN','EGF','NRG1','NRG2','NRG3','NRG4','TGFA','HBEGF','AREG','BTC','EPGN','EREG','EGFR','ERBB2','ERBB3','ERBB4']
	
genes_to_plot = []

for gene in fig_3B_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B_DotPlot_logScale.png', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.5, log=True)
sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B_DotPlot_linearScale.png', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.5, log=False)



fig_3B1_genes = ['LGR5','OLFM4']

genes_to_plot = []

for gene in fig_3B1_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B1_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, log=True)

fig_3B2_genes = ['FABP2','SI','DPP4']

genes_to_plot = []

for gene in fig_3B2_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B2_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, log=True)


fig_3B3_genes = ['F3','NPY','ACTA2','TAGLN']

genes_to_plot = []

for gene in fig_3B3_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B3_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, log=True)




fig_3B4_genes = ['EGF','NRG1','NRG2','NRG3','NRG4','TGFA','HBEGF','AREG','BTC','EPGN','EREG']

genes_to_plot = []

for gene in fig_3B4_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B4_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.15, log=True)



fig_3B5_genes = ['EGFR','ERBB2','ERBB3','ERBB4']

genes_to_plot = []

for gene in fig_3B5_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B5_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, log=True)


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

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.umap(adata, color=genes_to_plot, save = '_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)


fig_genes = ['EPCAM','VIM','ACTA2','EGF','EGFR','ERBB2','ERBB3','ERBB4','NRG1','RSPO1','RSPO2','RSPO3','WNT2','WNT2B','WNT3','DLL1','F3','NPY','GPX3']

genes_to_plot = []

for gene in fig_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.umap(adata, color=genes_to_plot, save = '_fig2_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)
sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save='_figure2_DotPlots.png', standard_scale='var', show=False, color_map=my_dot_cmap, dendrogram=False)

#sc.pl.tsne(adata, color=genes_to_plot, save = '_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)

fig_1D_genes = ['SHISA3','FIBIN','RGS5','PDGFRB','ANO1','KIT','ACTA2','TAGLN','PDGFRA','DLL1','F3','NPY','GPX3']

genes_to_plot = []

for gene in fig_1D_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, genes_to_plot, color_map = my_feature_cmap, groupby='louvain', var_group_positions=[(0,1),(2,3),(4,5),(6,7),(8,12)], var_group_labels=['Fibroblasts','Vasc. SMCs','ICCs','SMCs','Submucosal'], var_group_rotation=45, use_raw=True, log=True, dendrogram=True, expression_cutoff=expression_cutoff, mean_only_expressed=True, show=False, save='_fig_1D.png')
sc.pl.umap(adata, color=genes_to_plot, save = '_fig1D_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)


fig_2A_genes = ['ACTA2','TAGLN','DLL1','F3','NPY','GPX3']

genes_to_plot = []

for gene in fig_2A_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='age', use_raw=True, log=True, mean_only_expressed=True, expression_cutoff=1.0, save = '_fig2A_DotPlot.png', standard_scale='var', smallest_dot=0, show = False, color_map=my_dot_cmap, dendrogram=False)

sc.pl.matrixplot(adata, var_names=genes_to_plot, groupby='age', use_raw=True, log=False, save = '_fig2A_MatrixPlot.png', show = False)

sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig2B_DotPlot.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True)

sc.pl.umap(adata, color=genes_to_plot, save = '_fig2B_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)

adata.write('./data/Processed.concatenated.anndata.h5ad')


#sc.pl.scatter(adata, x='FABP2', y='EGF', use_raw=True, show=False, save='_FABP2-EGF_scatter.png')

#sc.pl.scatter(adata, x='EPCAM', y='VIM', use_raw=True, show=False, save='_EPCAM-VIM_EGF_scatter.png')













'''
print('Starting cell correlation calculations')
print('Creating dataframe...')
#sc.pp.subsample(adata, n_obs=5000, copy=False)
adata_df = pd.DataFrame(data=adata.X.transpose(), index=adata.var_names, columns=adata.obs_names)
print('Calculating correlation matrix')
adata_df_cm = adata_df.corr(method='pearson', min_periods=1)

annot_df = pd.DataFrame({'louvain':adata.obs['louvain'], 'age':adata.obs['age']})

cell_age = annot_df.pop("age")
lut = dict(zip(cell_age.unique(), greatestPalette))
row_colors = cell_age.map(lut)

sns.set(font='sans-serif', font_scale=1)
g = sns.clustermap(adata_df_cm, row_cluster=True, col_cluster=True, figsize=(40, 40), cmap="RdBu", row_colors=row_colors, col_colors=row_colors)
g.savefig('./figures/cell_correlation_matrix_heatmap.png')

print('Saving matrix to file')
adata_df_cm.to_csv('./figures/adata.corr_matrix.csv')
'''
















'''
adata = sc.read_h5ad('./data/Processed.concatenated.anndata.h5ad')
#adata.raw = sc.read_h5ad('./data/Filtered.concatenated.anndata.h5ad')

adata.raw = sc.read_h5ad('./data/Raw.concatenated.anndata.h5ad')
'''






print('Checking for expression of genes of interest\n')
expressed_dict = dict()

for gene in adata.raw.var_names.values.tolist():
	if gene not in expressed_dict:
		expressed_dict[str(gene)] = 1
'''


new_markers = ['PITX1','DCN','SNAI2','LTBP4','CALM2','FBLN1','LINC01082','MFAP4','BMP4','VCAN','COL1A1','SPARC','MEST','PTN','COL1A2',
'COL3A1','IGF2','FOS','LGALS1','ITM2C','FHL2','TSC22D1','SPARCL1','TPM1','ACTG2','SFRP1','CXCL12','MYL9','ACTA2','CALD1',
'KRT19','SPINK1','EPCAM','VIM','KRT18','SPINT2','KRT8','GSTP1','AGR2','CLDN6','ELAVL4','TUBB2B','MAP1B','MLLT11','TUBA1A',
'PCSK1N','TAGLN3','DPYSL3','KIF21A','CHRNA3','PLP1','AP1S2','S100B','EDNRB','ERBB3','PTPRZ1','METRN','CDH6','HMGB2','TUBA1B',
'H2AFZ','NUSAP1','HMGB1','KIAA0101','UBE2C','HMGN2','H2AFX','SMC4','MYH11','TAGLN','MYLK','TPM2','MYL6','DSTN','TGFBI','HMGA2',
'NKX2-5','WT1','TMSB4X','MDK','CRABP1','TLX1','BGN','TMSB10','B2M','CYBA','AIF1','ARHGDIB','TYROBP','LAPTM5','LST1','NGFRAP1',
'CD74','RAMP2','GNG11','CLDN5','ESAM','PRCP','EGFL7','ECSCR','S100A16','CDH5','PLVAP','FNDC1','COL14A1','CD81','TCF21','OGN',
'SHISA3','FN1','ZEB2','FIBIN','C1QTNF3','CXCL14','CTSC','NID1','IGFBP7','PDLIM1','F2R','ADAM28','COL6A3','COL18A1','HOXB5',
'BEX1','TCF4','MALAT1','S100A10','MEG3','ANXA13','CLDN3','PDGFRA','COL6A2','F3','COL6A1','TMEM176B','HAPLN1','ACKR3','EDIL3',
'NBL1','TOP2A','MKI67','CORO1A','RBP1','ARPC1B','HLA-E']


genes_to_plot = []

for gene in new_markers:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

#sc.pl.umap(adata, color=genes_to_plot, save = '_new_markers_featureplots1.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)

sc.pl.matrixplot(adata, genes_to_plot, cmap = my_feature_cmap, groupby='age', use_raw=True, log=True, dendrogram=False, standard_scale='var', show=False, save='_new_markers1.png')
sc.pl.dotplot(adata, genes_to_plot, color_map = my_feature_cmap, dot_max=0.75, groupby='age', use_raw=True, log=True, dendrogram=False, expression_cutoff=expression_cutoff, mean_only_expressed=True, standard_scale='var', show=False, save='_new_markers1.png')


sc.pl.matrixplot(adata, genes_to_plot, cmap = my_feature_cmap, groupby='louvain', use_raw=True, log=True, dendrogram=True, standard_scale='var', show=False, save='_louv_new_markers1.png')
sc.pl.dotplot(adata, genes_to_plot, color_map = my_feature_cmap, dot_max=0.75, groupby='louvain', use_raw=True, log=True, dendrogram=True, expression_cutoff=expression_cutoff, mean_only_expressed=True, standard_scale='var', show=False, save='_louv_new_markers1.png')


new_markers = ['SRGN','TGFB1I1','OLFML2B','FLNA','SLC15A3','POSTN','DBI','COL20A1',
'CDH19','IGFBP4','MADCAM1','A2M','ARHGAP29','IGFBP5','GUCY1A3','FABP5','TMEM176A','EMID1','ARPC3','VAMP8','PHOX2B','HAND2',
'NNAT','HAND2-AS1','LMNA','SEPT7','CCDC85B','TIE1','GPC3','ADAMDEC1','SHISA2','NKX2-3','DPT','ELN','DLL1','FRZB','NSG1',
'PFN1','CLDN4','PHGR1','ELF3','C19orf77','APOA1','CD9','S100A11','NELL2','UCHL1','CAV1','TUBB','STMN1','H2AFV','LUM',
'NDUFA4','IFITM3','CYGB','HLA-DRA','HLA-DRB1','HLA-DPA1','HLA-DPB1','HLA-DMA','RNASET2','HLA-DRB5','HLA-DMB','SLC7A7',
'RBP2','SERPINA1','FABP2','AMN','MUC13','APOB','PRAP1','RPL36','TSPAN8','RPS3','FOXF1','PTPRCAP','EVL','RAC2','KLRB1',
'CD52','LCK','ACTB','TPT1','MARCKSL1','AC011526.1','FKBP1A','CRIP2','RPL13A','RPL6','RPL10A','HNRNPA1','RPL15','RPL7',
'CST3','BMP5','ID3','CD7','MYL12A','FABP1','CDHR5','LGALS4','CLU','CYSTM1','TYMS','SPON2','CD37','CD79A','CD63','CD79B',
'VPREB3','LTB','ANXA2','CLEC14A','SRPX','HLA-B','RPLP1','CD69','LGALS3','SAT1','C7','KCNN3','ATP1B1','CNN3','RAMP1',
'TSHZ2','CYR61','MEIS2','SYT1','LMO4','S100A13','DLK1','PLAGL1','COX7A1','DES','CDK1','ALDH1A1','LGI4','NRXN1','APP',
'LAMP5','RPL5','GPX3','HLA-DQB1','HLA-DQA1','APOA4','ANPEP','STARD10','SEPW1','ENHO','RGS10','TINAGL1','HES4','PDGFRB',
'MGP','NOTCH3','COL4A2','HIGD1B','COL4A1']

genes_to_plot = []

for gene in new_markers:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

#sc.pl.umap(adata, color=genes_to_plot, save = '_new_markers2_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)

sc.pl.matrixplot(adata, genes_to_plot, cmap = my_feature_cmap, groupby='age', use_raw=True, log=False, dendrogram=False, standard_scale='var', show=False, save='_new_markers2.png')
sc.pl.dotplot(adata, genes_to_plot, color_map = my_feature_cmap, dot_max=0.75, groupby='age', use_raw=True, log=False, dendrogram=False, expression_cutoff=expression_cutoff, mean_only_expressed=True, standard_scale='var', show=False, save='_new_markers2.png')

sc.pl.matrixplot(adata, genes_to_plot, cmap = my_feature_cmap, groupby='louvain', use_raw=True, log=True, dendrogram=True, standard_scale='var', show=False, save='_louv_new_markers2.png')
sc.pl.dotplot(adata, genes_to_plot, color_map = my_feature_cmap, dot_max=0.75, groupby='louvain', use_raw=True, log=False, dendrogram=True, expression_cutoff=expression_cutoff, mean_only_expressed=True, standard_scale='var', show=False, save='_louv_new_markers2.png')
'''










fig_1C_genes = ['S100B','PLP1','STMN2','ELAVL4','CDH5','KDR','ECSCR','CLDN5','COL1A2','COL1A2','DCN','ACTA2','TAGLN','ACTG2','MYLK','EPCAM','CDH1','CDX2','CLDN4','PTPRC','HLA-DRA','ARHGDIB','CORO1A']

genes_to_plot = []

for gene in fig_1C_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.dotplot(adata, genes_to_plot, color_map = my_feature_cmap, groupby='louvain', var_group_positions=[(0,3),(4,7),(8,15),(16,19),(20,23)], var_group_labels=['ENS','Endothelial','Mesenchymal','Epithelial','Immune'], var_group_rotation=45, use_raw=True, log=True, dendrogram=True, expression_cutoff=expression_cutoff, mean_only_expressed=True, standard_scale='var', show=False, save='_fig_1C.png')





hand_picked_markers = ['DLL1','F3','NPY','GPX3','GJA1','VEGFA','VEGFB','VEGFC','VEGFD','SPN','POU5F1','FOXL1','FOXF1','PDGFRA','GLI1','KIT']

genes_to_plot = []

for gene in hand_picked_markers:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.umap(adata, color=genes_to_plot, save = '_forKate_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)



hashing_genes = ['EPCAM','VIM','COL1A1','DLL1','ATP1B3','B2M']

genes_to_plot = []

for gene in hashing_genes:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')

sc.pl.umap(adata, color=genes_to_plot, save = '_hashing_genes_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)



'''
print('Drawing line plots')
for i in range(len(genes_to_plot)):
	fig = plt.figure(dpi=80, figsize=(18,6))
	df = pd.DataFrame({'umap_1':adata.obsm['X_umap'][:,0],'umap_2':adata.obsm['X_umap'][:,1], 'louvain':adata.obs['louvain'].values, 'age':adata.obs['age'].values})
	df['louvain'] = 'clust_' + df['louvain'].astype(str)
	ax = fig.add_subplot(1,1,1)
	df[genes_to_plot[i]] = adata.raw[:, genes_to_plot[i]].X
	sns.lineplot(x="age", y=genes_to_plot[i], data=df, markers=True, ax=ax, ci=80)
	ax.set_xlabel('age')
	ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
	fig.savefig(''.join(['./figures/variance_plots_handpicked/lineplot_', genes_to_plot[i], '.png']))


top_new_markers = ['CDH5','ECSCR','PHOX2B','TYROBP','KLRB1','ESAM','CLDN5','PLVAP','CDH19','MADCAM1',
'SRGN','VPREB3','AC011526.1','CD79A','TIE1','HLA-DRB5','S100B','MYH11','ELAVL4',
'HLA-DQB1','LAMP5','DES','LAPTM5','PLP1','HLA-DQA1','HLA-DPA1','LCK','HLA-DRB1',
'APOA4','HIGD1B','ACTG2','COL20A1','LST1','HLA-DRA','TUBB2B','APOA1','CD7','CD79B',
'PTPRZ1','CLDN6','TLX1','HLA-DPB1','CORO1A','NRXN1','TAGLN','HLA-DMB','CD52','HLA-DMA',
'BEX1','CD74','PTPRCAP','KIF21A','RBP2','FABP2','PRAP1','TAGLN3','PCSK1N','CLEC14A',
'ARHGDIB','CHRNA3','LGI4','AIF1','CXCL14','MUC13','SERPINA1','PHGR1','TOP2A','HAPLN1',
'MYLK','KRT19','SPINK1','LTB','TSPAN8','F3','ACTA2','CLDN3','C19orf77','EGFL7',
'SLC15A3','FABP1','HAND2-AS1','NUSAP1','ELF3','SYT1','AMN','POSTN','NELL2','UCHL1',
'EPCAM','AGR2','APOB','CDK1','CDHR5','ENHO','A2M','SLC7A7','UBE2C','LGALS4','TINAGL1',
'OLFML2B','ADAMDEC1','MYL9','KIAA0101','MAP1B','ALDH1A1','CD69','DPYSL3','C7','CD37',
'ARHGAP29','ANPEP','CDH6','HAND2','TPM2','VAMP8','RAC2','KRT8','CAV1','DPT','KCNN3',
'NSG1','CLDN4','RAMP2','RNASET2','HOXB5','KRT18','CST3','ERBB3','BMP5','MLLT11',
'DLK1','EDNRB','ACKR3','TPM1','MKI67','HMGB2','CLU','GNG11','NKX2-5','TYMS','LGALS3',
'SAT1','GPX3','FABP5','FRZB','CRIP2','METRN','ANXA13','C1QTNF3','WT1','NOTCH3',
'S100A16','H2AFX','ARPC1B','SPINT2','HLA-E','PRCP','DLL1','COL1A1','RAMP1','BMP4',
'CD9','IGFBP7','IGFBP4','TUBA1B','AP1S2','NBL1','ATP1B1','PDGFRA','ANO1','KIT']

genes_to_plot = []

for gene in top_new_markers:
	if gene in expressed_dict:
		genes_to_plot.append(gene)
	else:
		print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')

print('Plotting genes:', ', '.join(genes_to_plot),'\n')


sc.pl.matrixplot(adata, genes_to_plot, cmap = my_feature_cmap, groupby='age', use_raw=True, log=False, dendrogram=False, standard_scale='var', show=False, save='_top_new_markers.png')
sc.pl.dotplot(adata, genes_to_plot, color_map = my_feature_cmap, dot_max=0.75, groupby='age', use_raw=True, log=True, dendrogram=False, expression_cutoff=expression_cutoff, mean_only_expressed=True, standard_scale='var', show=False, save='_top_new_markers.png')

sc.pl.matrixplot(adata, genes_to_plot, cmap = my_feature_cmap, groupby='louvain', use_raw=True, log=False, dendrogram=True, standard_scale='var', show=False, save='_louv_new_markers2.png')
sc.pl.dotplot(adata, genes_to_plot, color_map = my_feature_cmap, dot_max=0.75, groupby='louvain', use_raw=True, log=True, dendrogram=True, expression_cutoff=expression_cutoff, mean_only_expressed=True, standard_scale='var', show=False, save='_louv_top_new_markers.png')



print('Drawing line plots')
for i in range(len(genes_to_plot)):
	fig = plt.figure(dpi=80, figsize=(18,6))
	df = pd.DataFrame({'umap_1':adata.obsm['X_umap'][:,0],'umap_2':adata.obsm['X_umap'][:,1], 'louvain':adata.obs['louvain'].values, 'age':adata.obs['age'].values})
	df['louvain'] = 'clust_' + df['louvain'].astype(str)
	ax = fig.add_subplot(1,1,1)
	df[genes_to_plot[i]] = adata.raw[:, genes_to_plot[i]].X
	sns.lineplot(x="age", y=genes_to_plot[i], data=df, hue='louvain', markers=True, ax=ax, ci=80)
	ax.set_xlabel('age')
	ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
	fig.savefig(''.join(['./figures/variance_plots_topScores/lineplot_', genes_to_plot[i], '.png']))

print('Drawing strip plots')
for i in range(len(genes_to_plot)):
	fig = plt.figure(dpi=80, figsize=(20,5))
	df = pd.DataFrame({'umap_1':adata.obsm['X_umap'][:,0],'umap_2':adata.obsm['X_umap'][:,1], 'louvain':adata.obs['louvain'].values, 'age':adata.obs['age'].values})
	ax = fig.add_subplot(1,1,1)
	sns.despine(bottom=True, left=True)
	df[genes_to_plot[i]] = adata.raw[:, genes_to_plot[i]].X
	sns.stripplot(x="age", y=genes_to_plot[i], hue="louvain", palette=greatestPalette, data=df, dodge=True, jitter=0.2, alpha=.5, zorder=1, ax=ax)
	ax.set_xlabel('age')
	ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
	fig.savefig(''.join(['./figures/variance_plots_topScores/stripplot_', genes_to_plot[i], '.png']))



print('Done')
'''


sc.tl.dendrogram(adata, 'louvain', n_pcs=num_pcs_use, use_raw=True, cor_method='pearson', linkage_method='complete', key_added='dendrogram_louvain')

sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', n_genes=50, use_raw=True)
	
sc.tl.filter_rank_genes_groups(adata, groupby='louvain', use_raw=True, log=True, key_added='rank_genes_groups_filtered', min_in_group_fraction=0.25, min_fold_change=1.25, max_out_group_fraction=0.25)
sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups_filtered', groupby='louvain', mean_only_expressed=True,  n_genes=6, save = '_markerDotPlots.png', show = False, color_map=my_dot_cmap, dendrogram=True)

mjc.write_marker_file(adata)

adata.write('./data/Processed.concatenated.anndata.h5ad')








































