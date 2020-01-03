import sys
import bbknn
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from pathlib import Path
import mjc_functions as mjc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 3			 # verbosity: errors (0), warnings (1), info (2), hints (3)
#sc.settings.set_figure_params(dpi_save=1200, dpi=1200)
sns.set(style="white", color_codes=True)

greatestPalette = ["#f44336","#265bcf","#36b3ba","#ffeb3b","#e91e63","#00cc92",
"#4caf50","#ffb65c","#9c27b0","#03a9f4","#43d61f","#ff9800","#673ab7","#cddc39",
"#81bdc5","#ff5722","#fcc9c5","#acb4e2","#2effea","#fffbd6","#f7abc5","#b1dafb",
"#b5deb6","#ffe79e","#d88ae5","#90dbfe","#d5e9be","#ffd699","#bca6e3","#70eeff",
"#edf3ba","#ffccbd"]

palestPalette = ["#fcc9c5","#acb4e2","#2effea","#fffbd6","#f7abc5","#b1dafb","#b5deb6","#ffe79e","#d88ae5","#90dbfe","#d5e9be","#ffd699","#bca6e3","#70eeff","#edf3ba","#ffccbd"]



#############################################################################
##         Flag to toggle rerunning the analysis or just plotting          ##
#############################################################################

rerun = False

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
# UMAP arguments
num_neighbors_use = 20
num_pcs_use = 9
umap_spread = 1
umap_min_dist = 0.5
maxiter = None
umap_gamma=1
random_state = 20120608

paga_init = True

dot_size = 25 # size of each point un UMAP scatters

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
genes_of_interest = ['EPCAM','VIM','CDX2','EGF','EGFR','ERBB2','ERBB3','ERBB4','NRG1','RSPO1','RSPO2','RSPO3','WNT2','WNT2B','WNT3','ACTA2','COL1A1','SOX17','T','TOP2A','WT1',
'CDH5','PECAM1','VWF','KDR','CD34','RGS5', 'CSPG4','ITGAM','PTPRC','HBB','STMN2',
'S100B','TUBB3','SPDEF','CHGA','LYZ','MUC2','MUC5','VIL1','ALPI','DPP4','LCT','SHH','IHH','DHH', 'HHIP',
'GLI1','GLI2','SMO','PTCH1','LGR5','OLFM4','PDGFRA','ADAMDEC1','ADAM17','ADAM10',
'ADAM7','DLL1','F3','NPY','GPX3','BMP4','ANO1','KIT','HAND2','NKX2-5','MSLN','WT1','UPK3B',
'HES1', 'HEY1','ID1','ID2','ID3','ID4','NOG','CHRD','GREM1','FOXF1','FOXF2','FOXL1',
'FOXL2','VEGFA','LOXL2','LAMC1','CYGB','FRZB','CTGF','CTSC','C1S','SYNPO2','EVA1B',
'ACKR3','XIST']

epi_cell_type_genes = ['VIM','EPCAM','CHD1','LGR5','OLFM4','MUC2','MUC5','CHGA','LYZ','DEF5A','DEF4A','DEF6A','VIL1','DPP4','FABP1','ALPI','SI','MGAM','SPDEF','SOX9','ATOH1','GFI1','NEUROG3','SPDEF','DCLK1']

fig_genes = ['EPCAM','VIM','ACTA2','EGF','EGFR','ERBB2','ERBB3','ERBB4','NRG1','RSPO1','RSPO2','RSPO3','WNT2','WNT2B','WNT3','DLL1','F3','NPY','GPX3']

y_chrom_genes = ['RPS4Y1','ZFY','TTTY23B','TTTY15','USP9Y','DDX3Y','NLGN4Y','KDM5D','XIST','TSIX','CYP19A1']

y_chrom_genes_only = ['RPS4Y1','ZFY','TTTY23B','TTTY15','USP9Y','DDX3Y','NLGN4Y','KDM5D']

androgen_genes_only = ['XIST','CYP19A1']

emilys_list = ['PGK1','CDH1','EPCAM','CDX2','VIL1','SLC2A5','ALPI','MUC2','MUC5AC','MUC5B','TFF3','LYZ','DEFA5','DEFA6','LGR5',
'TAGLN','COL1A1','COL2A1','VCL','VIM','ACTA2','PTPRC','CD163','CD80','CD86','ARG1','CD14','MRC1','MS4A1','CCR5','CD4','CD8A',
'CD8B','CR2','TGFB1','IGF1','IL6','IL1B','IGF1R','FGF2','HIF1A','TNF','TGFBR2','AXL','GAS6','MERTK','CD44','KLF1','TLR4','MAPK14',
'MAPK1','MAPK3','RAF1','KRAS','AKT1','AKT2','AKT3','RHOA','ROCK1','ROCK2','SMAD1','SMAD2','SMAD3','SMAD4','SMAD5','SMAD6','SMAD7',
'SMAD9','TYK2','STAT3','STAT1','JAK2','HIF1A','MMP9','MTOR','SOCS1','SOCS3','SNAI1','TWIST1']

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








def process_adata(adata):
	# Assign score for gender based on expression of Y-chromosome genes
	sc.tl.score_genes(adata, y_chrom_genes_only, ctrl_size=50, gene_pool=None, n_bins=25, score_name='maleness', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, androgen_genes_only, ctrl_size=50, gene_pool=None, n_bins=25, score_name='femaleness', random_state=0, copy=False, use_raw=False)
	
	sc.pl.violin(adata, keys=['maleness', 'femaleness'], groupby='age', save='_gender_plot.png', show=False, ax=None)
	
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
	#sc.tl.umap(adata, init_pos='spectral', min_dist=umap_min_dist, maxiter=maxiter, spread=umap_spread, gamma=umap_gamma, random_state=random_state)
	
	## Run tSNE algorithm
	
	sc.tl.tsne(adata, n_pcs=num_pcs_use)
	
	## Run draw_graph to get a FA2 graph layout
	
	sc.tl.draw_graph(adata,layout='fa', init_pos='paga', scalingRatio=4.0)
	
	
	
	sc.pl.umap(adata, color='louvain', save = '_clusterIdentity_noEdge.png', show = False, legend_loc = 'on data', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	sc.pl.umap(adata, color=['louvain', 'age'], save = '_clusterIdentity_age.png', show = False, legend_loc = 'right margin', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	sc.pl.umap(adata, color='age', save = '_age.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='sex', save = '_sex.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='sampleName', save = '_sample.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color=['n_genes','n_counts','percent_mito'], save = '_stats.png', show = False, edges = False, cmap = my_feature_cmap, size = dot_size+10)
	
	sc.pl.tsne(adata, color='louvain', save = '_clusterIdentity_noEdge.png', show = False, legend_loc = 'on data', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	sc.pl.tsne(adata, color=['louvain', 'age'], save = '_clusterIdentity_age.png', show = False, legend_loc = 'right margin', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	
	sc.pl.draw_graph(adata, color='louvain', save = '_clusterIdentity_noEdge.png', show = False, legend_loc = 'on data', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	sc.pl.draw_graph(adata, color=['louvain', 'age'], save = '_clusterIdentity_age.png', show = False, legend_loc = 'right margin', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	
	sc.pl.paga(adata, color='louvain', save=''.join(['_', paga_layout, '_page.png']), show=False, threshold=threshold, node_size_scale=node_size_scale, node_size_power=0.9, layout=paga_layout)
	
	'''
	sc.tl.tsne(adata, n_pcs=num_pcs_use, use_rep='X_pca', perplexity=30, early_exaggeration=12, learning_rate=1000, random_state=random_state, use_fast_tsne=True, n_jobs=10, copy=False)
	
	sc.pl.tsne(adata, color='louvain', save = '_clusterIdentity_noEdge.png', show = False, legend_loc = 'on data', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	sc.pl.tsne(adata, color=['louvain', 'age'], save = '_clusterIdentity_age.png', show = False, legend_loc = 'right margin', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	sc.pl.tsne(adata, color='age', save = '_age.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95)
	sc.pl.tsne(adata, color='sex', save = '_sex.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95)
	sc.pl.tsne(adata, color='sampleName', save = '_sample.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95)
	sc.pl.tsne(adata, color=['n_genes','n_counts','percent_mito'], save = '_stats.png', show = False, edges = False, cmap = my_feature_cmap, size = dot_size+10)
	'''
	
	sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', n_genes=100, use_raw=True)
	#sc.tl.filter_rank_genes_groups(adata, groupby='louvain', use_raw=True, log=True, key_added='rank_genes_groups_filtered', min_in_group_fraction=0.05, min_fold_change=1, max_out_group_fraction=0.95)
	sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups', groupby='louvain', mean_only_expressed=True,  n_genes=6, save = '_markerDotPlots.png', show = False, color_map=my_dot_cmap, dendrogram=True)
	mjc.write_marker_file(adata, file_out=''.join([figure_dir, '/marker_output.csv']), n_genes=100)
	
	
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
	
	genes_to_plot = []
	for gene in epi_cell_type_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.umap(adata, color=genes_to_plot, save = '_epi_cell_types_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)
	
	
	genes_to_plot = []
	for gene in emilys_list:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.umap(adata, color=genes_to_plot, save = '_emilysGenes_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)
	
	
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
	
	fig_1D_genes = ['DCN','COL1A1','COL1A2','RGS5','PDGFRB','ANO1','KIT','ACTA2','TAGLN','PDGFRA','DLL1','F3','NPY','GPX3']
	genes_to_plot = []
	for gene in fig_1D_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.dotplot(adata, genes_to_plot, color_map = my_feature_cmap, groupby='louvain', var_group_positions=[(0,2),(3,4),(5,6),(7,8),(9,13)], var_group_labels=['Fibroblasts','Vasc. SMCs','ICCs','SMCs','Submucosal'], var_group_rotation=45, use_raw=True, log=True, dendrogram=True, expression_cutoff=expression_cutoff, mean_only_expressed=True, show=False, save='_fig_1D.png')
	sc.pl.dotplot(adata, genes_to_plot, color_map = my_feature_cmap, groupby='louvain', var_group_positions=[(0,2),(3,4),(5,6),(7,8),(9,13)], var_group_labels=['Fibroblasts','Vasc. SMCs','ICCs','SMCs','Submucosal'], var_group_rotation=45, use_raw=True, log=True, dendrogram=True, expression_cutoff=expression_cutoff, mean_only_expressed=True, show=False, save='_fig_1D.pdf')
	sc.pl.umap(adata, color=genes_to_plot, save = '_fig1D_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)
	
	genes_to_plot = []
	for gene in y_chrom_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.umap(adata, color=genes_to_plot, save = '_y_chrom_featurePlots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)
	#sc.pl.tsne(adata, color=genes_to_plot, save = '_y_chrom_featurePlots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)
	#sc.tl.dendrogram(adata, 'louvain', n_pcs=num_pcs_use, use_raw=True, cor_method='pearson', linkage_method='complete', key_added='dendrogram_louvain')
	sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', n_genes=50, use_raw=True)
	sc.tl.filter_rank_genes_groups(adata, groupby='louvain', use_raw=True, log=True, key_added='rank_genes_groups_filtered', min_in_group_fraction=0.25, min_fold_change=1.5, max_out_group_fraction=0.5)
	sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups_filtered', groupby='louvain', mean_only_expressed=True,  n_genes=10, save = '_markerDotPlots.png', show = False, color_map=my_dot_cmap, dendrogram=True)
	
	
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
	
	
	
	fig_3A_genes = ['ACTA2','TAGLN','F3','NPY','GPX3','WNT2B','RSPO2','RSPO3','NOG','CHRD','EGF']
	genes_to_plot = []
	for gene in fig_3A_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', use_raw=True, log=False, mean_only_expressed=True, save = '_fig3A_DotPlot.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.5)
	sc.pl.umap(adata, color=genes_to_plot, save = '_fig3A_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)
	
	crypt_SEC_genes = ['F3','DLL1','COL15A1','NRG1','CH25H','MMP11','CXCR4','CPM','BMP3','IGFBP5','ADAMDEC1','CTGF','CRY61','IGFBP3','HHIP','EFEMP1','NPY','CTCSC','NBEAL1','EIF5A','RPSAP58']
	genes_to_plot = []
	for gene in crypt_SEC_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', use_raw=True, log=False, mean_only_expressed=True, save = '_crypt_SEC_DotPlot.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True)
	sc.pl.umap(adata, color=genes_to_plot, save = '_crypt_SEC_featureplots.png', show = False, cmap = my_feature_cmap, size = dot_size*3, use_raw = True)
	
	
	fig_3B_genes = ['LGR5','OLFM4','FABP2','SI','DPP4','F3','NPY','ACTA2','TAGLN','NRG1','NRG2','NRG3','NRG4','TGFA','HBEGF','AREG','BTC','EPGN','EREG','EGFR','ERBB2','ERBB3','ERBB4']
	genes_to_plot = []
	for gene in fig_3B_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.25, log=True)
	sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B_DotPlot_linearScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.25, log=False)
	
	
	fig_3B1_genes = ['LGR5','OLFM4']
	genes_to_plot = []
	for gene in fig_3B1_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B1_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.25, log=True)
	
	fig_3B2_genes = ['FABP2','SI','DPP4']
	genes_to_plot = []
	for gene in fig_3B2_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B2_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.25, log=True)
	
	
	fig_3B3_genes = ['F3','NPY','ACTA2','TAGLN']
	genes_to_plot = []
	for gene in fig_3B3_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B3_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.25, log=True)
	
	
	
	fig_3B4_genes = ['NRG1','NRG2','NRG3','NRG4','TGFA','HBEGF','AREG','BTC','EPGN','EREG']
	genes_to_plot = []
	for gene in fig_3B4_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B4_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.25, log=True)
	
	
	fig_3B5_genes = ['EGFR','ERBB2','ERBB3','ERBB4']
	genes_to_plot = []
	for gene in fig_3B5_genes:
		if gene in expressed_dict:
			genes_to_plot.append(gene)
		else:
			print('Sorry,', gene, 'Is not expressed in this dataset or is invariable.\n')
	
	print('Plotting genes:', ', '.join(genes_to_plot),'\n')
	sc.pl.dotplot(adata, var_names=genes_to_plot, groupby='louvain', mean_only_expressed=True, save = '_fig3B5_DotPlot_logScale.png', standard_scale='var', show = False, color_map=my_dot_cmap, dendrogram=True, dot_max=0.25, log=True)
	
	return(adata)











tmp_adata = sc.read_h5ad('/mnt/black/scRNA-seq/scanpy/Scanpy_batch-corrected_fetal_intestine/data/Processed.concatenated.anndata.h5ad')
orig_adata = sc.read_h5ad('/mnt/black/scRNA-seq/scanpy/Scanpy_batch-corrected_fetal_intestine/data/Filtered.concatenated.anndata.h5ad')



#cells_to_keep = tmp_adata[tmp_adata.obs['louvain'].isin(['0', '1', '3', '2', '12', '6', '14', '17'])].obs.index.tolist()
init_adata = orig_adata


#init_adata = sc.read_h5ad('./data/Filtered.concatenated.anndata.h5ad')

init_adata.raw = init_adata

init_adata.obs['louvain'] = tmp_adata.obs['louvain']

init_adata.write('./data/Mesenchyme.concatenated.anndata.h5ad')

init_adata = sc.read_h5ad('./data/Mesenchyme.concatenated.anndata.h5ad')



init_adata[init_adata.obs['age'].isin(['47'])].write('./data/Early_filtered.concatenated.anndata.h5ad')

init_adata[init_adata.obs['age'].isin(['59', '72'])].write('./data/Mid_filtered.concatenated.anndata.h5ad')

init_adata[init_adata.obs['age'].isin(['80', '101', '122', '127', '132'])].write('./data/Late_filtered.concatenated.anndata.h5ad')


init_adata[init_adata.obs['louvain'].isin(['3','5','13'])].write('./data/all_epithelial_filtered.concatenated.anndata.h5ad')


init_adata[init_adata.obs['louvain'].isin(['0','1','2','4','7'])].write('./data/all_mesenchyme_filtered.concatenated.anndata.h5ad')

init_adata = sc.read_h5ad('./data/all_mesenchyme_filtered.concatenated.anndata.h5ad')

init_adata[init_adata.obs['age'].isin(['47'])].write('./data/Early-mesenchyme_filtered.concatenated.anndata.h5ad')

init_adata[init_adata.obs['age'].isin(['59', '72'])].write('./data/Mid-mesenchyme_filtered.concatenated.anndata.h5ad')

init_adata[init_adata.obs['age'].isin(['80', '101', '122', '127', '132'])].write('./data/Late-mesenchyme_filtered.concatenated.anndata.h5ad')




figure_dir = './figures/all_mesenchyme'
sc.settings.figdir = figure_dir
adata = sc.read_h5ad('./data/all_mesenchyme_filtered.concatenated.anndata.h5ad')
adata_all_mesenchyme = process_adata(adata)
adata_all_mesenchyme.write('./data/all_mesenchyme_processed.concatenated.anndata.h5ad')

ageFigumap = plt.figure(dpi=80, figsize=(24,7))
ax1 = ageFigumap.add_subplot(1,3,1)
ax2 = ageFigumap.add_subplot(1,3,2)
ax3 = ageFigumap.add_subplot(1,3,3)

sc.pl.umap(adata_all_mesenchyme[adata_all_mesenchyme.obs['age'].isin(['47'])], color='louvain', save = '_sub-early_louvain.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, ax=ax1)
sc.pl.umap(adata_all_mesenchyme[adata_all_mesenchyme.obs['age'].isin(['59', '72'])], color='louvain', save = '_sub-mid_louvain.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, ax=ax2)
sc.pl.umap(adata_all_mesenchyme[adata_all_mesenchyme.obs['age'].isin(['80', '101', '122', '127', '132'])], color='louvain', save = '_sub-late_louvain.png', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, ax=ax3)

ax1.set_title('Louvain - Early Stage')
ax2.set_title('Louvain - Mid Stages')
ax3.set_title('Louvain - Late Stages')

ageFigumap.savefig(''.join([figure_dir,'/UMAP_louvain_age_panels.png']))

'''

figure_dir = './figures/all_epithelial'
sc.settings.figdir = figure_dir
adata = sc.read_h5ad('./data/all_epithelial_filtered.concatenated.anndata.h5ad')
adata_epithelial = process_adata(adata)
adata_epithelial.write('./data/All_epithelial_processed.anndata.h5ad')



figure_dir = './figures/early_mesenchyme'
sc.settings.figdir = figure_dir
adata = sc.read_h5ad('./data/Early-mesenchyme_filtered.concatenated.anndata.h5ad')
adata_early = process_adata(adata)
adata_early.write('./data/Early_mesenchyme_filtered.concatenated.anndata.h5ad')

figure_dir = './figures/mid_mesenchyme'
sc.settings.figdir = figure_dir
adata = sc.read_h5ad('./data/Mid-mesenchyme_filtered.concatenated.anndata.h5ad')
adata_mid = process_adata(adata)
adata_mid.write('./data/Mid_mesenchyme_filtered.concatenated.anndata.h5ad')

figure_dir = './figures/late_mesenchyme'
sc.settings.figdir = figure_dir
adata = sc.read_h5ad('./data/Late-mesenchyme_filtered.concatenated.anndata.h5ad')
adata_late = process_adata(adata)
adata_late.write('./data/Late_mesenchyme_filtered.concatenated.anndata.h5ad')


subepi_cells = adata_late[adata_late.obs['louvain']=='3'].obs.index.tolist()
subepi_adata = init_adata[init_adata.obs.index.isin(subepi_cells)]
subepi_adata.write('./data/Late_subepithelial_filtered.concatenated.anndata.h5ad')

figure_dir = './figures/late_subepithelial'
sc.settings.figdir = figure_dir
adata = sc.read_h5ad('./data/Late_subepithelial_filtered.concatenated.anndata.h5ad')
adata_late_subepi = process_adata(adata)
adata_late_subepi.write('./data/Late_subepithelial_filtered.concatenated.anndata.h5ad')




figure_dir = './figures/early'
sc.settings.figdir = figure_dir
adata = sc.read_h5ad('./data/Early_filtered.concatenated.anndata.h5ad')
adata_early = process_adata(adata)
adata_early.write('./data/Early_processed.concatenated.anndata.h5ad')

figure_dir = './figures/mid'
sc.settings.figdir = figure_dir
adata = sc.read_h5ad('./data/Mid_filtered.concatenated.anndata.h5ad')
adata_mid = process_adata(adata)
adata_mid.write('./data/Mid_processed.concatenated.anndata.h5ad')

figure_dir = './figures/late'
sc.settings.figdir = figure_dir
adata = sc.read_h5ad('./data/Late_filtered.concatenated.anndata.h5ad')
adata_late = process_adata(adata)
adata_late.write('./data/Late_processed.concatenated.anndata.h5ad')











louvFigumap = plt.figure(dpi=80, figsize=(24,7))
ax1 = louvFigumap.add_subplot(1,3,1)
ax2 = louvFigumap.add_subplot(1,3,2)
ax3	= louvFigumap.add_subplot(1,3,3)

sc.pl.umap(adata_early, color='louvain', show = False, legend_loc = 'on data', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6, ax=ax1)
sc.pl.umap(adata_mid, color='louvain', show = False, legend_loc = 'on data', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6, ax=ax2)
sc.pl.umap(adata_late, color='louvain', show = False, legend_loc = 'on data', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6, ax=ax3)

ax1.set_title('Louvain clusters: Early timepoint (47 days)')
ax2.set_title('Louvain clusters: Mid timepoint (59-72 days)')
ax3.set_title('Louvain clusters: Late timepoint (80-132 days)')
	
louvFigumap.savefig(''.join([figure_dir, '/UMAP_clusterIdentity_age_panels.png']))



ageFigumap = plt.figure(dpi=80, figsize=(24,7))
ax1 = ageFigumap.add_subplot(1,3,1)
ax2 = ageFigumap.add_subplot(1,3,2)
ax3	= ageFigumap.add_subplot(1,3,3)

sc.pl.umap(adata_early, color='age', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6, ax=ax1)
sc.pl.umap(adata_mid, color='age', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6, ax=ax2)
sc.pl.umap(adata_late, color='age', show = False, legend_loc = 'right margin', edges = False, size = dot_size, palette = greatestPalette, alpha = 0.95, legend_fontsize=6, ax=ax3)

ax1.set_title('Age: Early timepoint (47 days)')
ax2.set_title('Age: Mid timepoint (59-72 days)')
ax3.set_title('Age: Late timepoint (80-132 days)')
	
ageFigumap.savefig(''.join([figure_dir, '/UMAP_early-mid-late_age_panels.png']))

'''


























