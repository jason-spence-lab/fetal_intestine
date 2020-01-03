import bbknn
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from pathlib import Path
import mjc_functions as mjc
import matplotlib.pyplot as plt
sc.settings.verbosity = 3			 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi_save=800, dpi=800)
import warnings
warnings.filterwarnings('ignore')

greatestPalette = ["#f44336","#265bcf","#36b3ba","#ffeb3b","#e91e63","#00cc92","#4caf50","#ffb65c","#9c27b0","#03a9f4","#43d61f","#ff9800","#673ab7","#cddc39","#81bdc5","#ff5722"]
palestPalette = ["#fcc9c5","#acb4e2","#2effea","#fffbd6","#f7abc5","#b1dafb","#b5deb6","#ffe79e","#d88ae5","#90dbfe","#d5e9be","#ffd699","#bca6e3","#70eeff","#edf3ba","#ffccbd"]

figure_dir = './figures/E1N100'
sc.settings.figdir = figure_dir

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
sample_list = ['150-3']


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
	sc.pp.filter_genes(adata, min_cells=10)
	
	print('\nDoing initial filtering...\nKeeping', len(adata.obs_names), 'cells and', len(adata.var_names), 'genes.\n')
	
	cell_cycle_genes = [x.strip() for x in open('/mnt/black/scRNA-seq/regev_lab_cell_cycle_genes.txt')]
	
	s_genes = cell_cycle_genes[:43]
	g2m_genes = cell_cycle_genes[43:]
	cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
	
	sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
	
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
	adata = adata[adata.obs['n_genes'] < 9000, :]   # Keep cells with less than 5000 genes to remove most doublets
	adata = adata[adata.obs['n_counts'] < 60000, :] # Keep cells with less than 15000 UMIs to catch a few remaining doublets
	adata = adata[adata.obs['percent_mito'] < 0.5, :]   # Keep cells with less than 0.1 mito/genomic gene ratio
	sc.pp.filter_genes(adata, min_cells=50)	# Refilter genes to get rid of genes that are only in a tiny number of cells
	
	print('\nDoing final filtering...\nKeeping', len(adata.obs_names), 'cells and', len(adata.var_names), 'genes.\n')
	
	open('./data/Final_filtered_gene_list.txt', 'w').write('\n'.join(adata.var_names.tolist()))
	
	sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
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
	
	clust_1_genes = ['APOC3','SEPP1','APOA4','APOA1','PRAP1','SERPINA1','B2M','CAPN3','FTH1','FABP2','APOB','SLC7A7','IL32','SLC40A1','ITM2B','MUC13','C19orf77','NEAT1','VAMP8','LGMN','SAT1','MAMDC4','AMN','CTSA','CDHR5','RHOC','SAT2','FAM3C','GRN','C8G','ID2','CTSH','GCHFR','TDP2','ASAH1','DAB2','PCSK1N','CLTB','MYL9','PHGR1','ATOX1','NEU1','FBP1','ETHE1','RTN4','ACSL5','CIDEB','NPC2','HLA-C','SLC39A4','FOLH1','LPGAT1','HLA-B','G0S2','CTSB','CD63','MAX','MFI2','KHK','GNS','DHRS7','PEPD','RARRES1','SLC5A12','SLC46A3','MME','AGPAT2','TMEM176A','SLC25A5','FN3K','FAM132A','ALDOB','NR0B2','CD74','HSD17B11','BTNL3','ALPI','CST3','REEP3','SLC15A1','ADA','RBP2','MAF','GPX4','TMEM92','ANPEP','SPG21','FBXO2','PLAC8','OCIAD2','TNFSF10','MXRA8','COX7A2L','SLC6A19','C19orf69','NR1H4','GLRX','FABP1']
	clust_2_genes = ['GSTA1','RBP2','FTL','ANXA13','MTTP','S100A14','S100A10','EEF1B2','TPT1','LGALS3','ENPP7','TTR','AADAC','IGF2','ANXA4','GSTA2','CALM1','HEBP1','OAT','DBI','CD68','FCGRT','ANPEP','SLC26A3','QPRT','GUK1','DGAT1','PBLD','AFP','SLC9A3R1','CTSA','TMEM256','SCT','NDRG1','REEP6','GSTO1','MALL','C19orf33','SNX3','APOA1','FABP2','DHRS11','SLC7A7','NACA','SI','MAF','TMEM141','PEPD','CRYL1','SLC2A2','ASS1','CYP2W1','ACE2','GNB2L1','RARRES1','ANXA2','ATP1B3','SLC37A4','ATP5G3','KRT8','AIG1','SUCLG1','SLC25A5','SAT2','DNASE1','FBXO2','CEBPA','MYL12B','HSD17B2','MEP1A','FBP1','MISP','MAP2K2','HADH','MGST3','CFL1','ATP1A1','ATP5E','RBP2']
	clust_3_genes = ['FTL','SLC7A7','FTH1','CTSA','AMN','MAMDC4','SLC26A3','RARRES1','S100A14','SERPINA1','APOA4','NPC2','FABP2','MYL9','C19orf77','MUC13','ALDOB','SEPP1','APOA1','APOC3','OAT','SAT2','GUK1','PRAP1','PEPD','B2M','ITM2B','C19orf33','ASS1','FCGRT','ASAH1','CDHR5','CD68','G0S2','FAM3C','SLC25A5','GCHFR','ID2','FBP1','SCT','FABP1','KHK','DHRS7','SMIM1','TTR','FOLH1','APOB','MS4A8','SLC5A12','CD63','PCSK1N','RHOC','ATP1B3','ESPN','ENPP7','MAF','COX7A2L','COX7C','ETHE1','TM4SF20','IL18','CIDEB','MTTP','IL32','SMLR1','FBXO2','F10','CBR1','SLC9A3R1','GRN','NR0B2','GOLT1A','CTSH','TDP2','ANPEP','SLC6A19','VAMP8','ANXA4','CST3','HSD17B11','PRR13','TMEM92','LGALS2','ACSL5','ADA','SLC40A1','PHPT1','SLC2A2','NEAT1','ATP5E','SCPEP1','SLC15A1','HADH','LGMN','TMEM176A','PLAC8','ATP6V1F','HSPD1']
	clust_4_genes = ['HMGA1','TUBA1B','AGR2','HSP90AA1','YBX1','TXN','PTMA','MT1G','MT1E','HSP90AB1','PPIA','RAN','HMGB1','TKT','H2AFZ','CCL25','NCL','ACTG1','HNRNPA2B1','MT2A','EIF5A','SERBP1','FABP5','ENO1','TUBB','LDHB','MT1H','GAPDH','EIF4A1','C1QBP','STMN1','IGFBP2','NPM1','RANBP1','MT1X','SNRPB','NME1','MT1F','HSP90B1','ATP5B','GNAS','SET','HNRNPA1','HSPE1','TMSB10','HMGN2','S100A10','NASP','SNRPF','TUFM','ODC1','PA2G4','IMPDH2','TSPAN8','FBL','HNRNPD','SNRPD1','MARCKSL1','MLEC','CCT2','HNRNPM','AHCY','HMGB2','PRMT1','EEF1B2','HNRNPA3','HIST1H4C','LSM5','MZT2B','NOP56','MT-ND1','MDH2','HSPA8','NUCKS1','KRT19','PEBP1','CYC1','SRSF3','DCTPP1','PHB','CCT8','TSPAN8']
	clust_5_genes = ['HSP90AB1','GNB2L1','AGR2','SLC12A2','ELF3','IMPDH2','NPM1','JUN','PTMA','EEF1A1','PABPC1','MT-CO3','HNRNPA1','BTF3','MYC','BEX1','MT-CO1','TMSB10','MT-ND4','ZFP36L2','MT-CYB','ASCL2','EEF2','HMGCS2','NACA','NAP1L1','IER2','MARCKSL1','ZFAS1','TXN','MT-CO2','HES1','CLU','EDN1','S100A10']
	clust_6_genes = ['GSTA1','CCL25','GNB2L1','TSPAN8','DMBT1','ACTG1','REEP6','NACA','KRT8','SLC25A6','FABP1','CLDN3','EEF1A1','EEF1B2','GAPDH','SULT1E1','CYP2W1','CPS1','RBP1','CES2','SNHG8','DNASE1','LGALS2','AGR2','ZFAS1','LGALS3','MT-CO3','CLU','MT1G','ANXA13','HNRNPA1','MT1E','NPM1','MT-CO2','DBI','CLIC1','MT-CO1','TUBB']
	clust_7_genes = ['NAP1L1','STMN1','HMGA1','HNRNPA1','MARCKSL1','MDK','TUBA1B','TXN','LDHB','TMSB10','FGB','PTMA','NPM1','YBX1','HMGB1','H2AFZ','ACTG1','HSP90AB1','CKB','GPC3','ALDH1A1','SNRPE','RAN','PABPC1','PPDPF','FABP5','REG4','H3F3A','AGR2','NME4','KRT19','RANBP1','GAPDH','HSP90AA1','CCNB1IP1','BTF3','ENO1','ODC1','PPIA','HMGN1','IMPDH2','HMGN2','PKM','HBG2','ACTB','HNRNPA0','EIF5A','EEF2','SUMO2','TUBB2B','UBA52','HMGCS2','PRMT1','SNRPD1','HNRNPA2B1','FBL','YWHAQ','HMGB2','EIF4A1','ILF2','SRSF3','HMGB3','IDH2','NGFRAP1','RBM3','EEF1A1','HBA2','SRP9','FBLN1','HSPD1','SRSF2','HDAC2','COL3A1']
	clust_8_genes = ['LGALS1','COL1A2','VIM','COL1A1','MFAP4','SPARC','DCN','CALD1','COL6A2','PTN','TCF21','LTBP4','MEG3','FBLN1','LUM','VCAN','MDK','FSTL1','IGFBP5','TPM2','MMP2','COL6A3','EMILIN1','CXCL12','OGN','SNAI2','COL5A2','TUBA1A','ZEB2','FHL1','SPON2','TCF4','IFITM3','SPARCL1','PCOLCE','COL6A1','RARRES2','7-Sep','CYR61','C11orf96','PMP22','SHISA3','OLFML3','NGFRAP1','COL5A1','SOX4','NREP','LINC01082','FOS','FN1','FOSB','NID1','MARCKS','EMP3','SERPINH1','TSHZ2','NFIA','PTMS','MMP23B','MDFI','CLEC11A','FXYD6','IGFBP7','PDGFRA','RGS10','MEST','EFEMP2','ADAMDEC1','TPM1','CXCL14','SDC2','H3F3B','CNN3','ELN','IGFBP4','FOXF1','ZFP36L1','TIMP1','THY1','NFIB','NAP1L1','MALAT1','EGR1','MFAP2','C12orf57','EPB41L2','11-Sep','JUNB','C1QTNF3','NKX2-3','JUN','RBP1','HSPB1','CD81','CCL2','TUBB','TGFB1I1','LAPTM4A','CALU','S100A6']
	clust_9_genes = ['LGALS4','GSN','SH3BGRL3','TFF3','HEPACAM2','HSPB1','MDK','LYZ','PPDPF','S100A11','FXYD3','SELM','FCGBP','TMSB4X','CLCA1','KRT19','RNASE1','GUCA2A','MYL6','HPCAL1','NEURL1','TPM1','SPINK1','TPD52','HES6','GUCA2B','SPIB','SERF2','ITLN1','FRZB','KRT8','LRRC26','MUC2','BCAS1','H3F3B','KLF4','PLP2','REP15','CD9','JUN','STARD10','ST6GALNAC1','YWHAZ','QSOX1','AGR3','MT2A','IER2','CALM2','GPX2','AGR2','KRT20','TMSB10','MGLL','FOS','KLK1','CDC42EP5','DLL1','BTG2','CTSE','HMGN2','TXNIP','WFDC2','MARCKSL1','LINC00261','FOXA3','CEACAM6','UCP2','ALDH2','XBP1','PRSS1','TCEA3','EGR1','ELF3','H3F3A','CA7','DYNLL1','RBPJ','RAB11FIP1','SPATS2L','TSPAN1','TSTA3','SLC38A2','C12orf57','GMDS','FOSB','RASSF6','SSR4','CEACAM5','DYRK4','WSB1','KIAA1324','SELK','ENHO','LPCAT4','BEST4','ABCA4','FOXP1','CFTR','SYTL2','CHGA']
	clust_10_genes = ['CPE','SCGN','NEUROD1','PCSK1N','TUBA1A','TM4SF4','SCG2','FEV','BEX1','CHGB','CRYBA2','PCSK1','CADPS','INSM1','SSR4','KCTD12','GC','SMIM6','NGFRAP1','NPDC1','RGS2','C9orf16','PTMS','SCG3','PAM','HEPACAM2','LY6H','HOPX','AES','EID1','SCG5','UCP2','BEX2','SEC11C','H3F3B','DDX5','GABARAPL2','GNAS','MAP1B','GAD2','KIAA1244','RAB3B','S100A6','QPCT','VWA5B2','PROX1','STMN2','NKX2-2','RAB26','MIR7-3HG','ALDH1A1','CBX6','RFX6','GTF2I','NLRP1','STMN1','RIMBP2','DNAJC12','PKM','TUSC3','DSP','MAP1LC3A','AK1','VAMP2','QDPR','C1QL1','CNIH2','LRRFIP1','CACNA1A','PPDPF','SCT','APLP1','NOVA1','TTR','SYP','MYL6','LINC00261','CACNA2D1','MARCKSL1','FXYD3','ABCC8','MARCKS','ZKSCAN1','PCBP4','KIAA1324','PFN2','TBCB','IDS','SYT13','SSTR2','GHRL','SNAP25','VTN','TUBA4A','MDK','EGR1','FOXA2','GCH1','C10orf10','APOB']
	clust_11_genes = ['MT-ND3','MTTP','TTR','SI','MT-ATP6','MT-CO2','MT-ND1','MAF','AFP','MT-ND5','CDHR5','MUC13','CDKN1C','DGAT1','ENPP7','MT-ND4','APOA4','ACE2','MT-CYB','LPGAT1','PNCK','MT-CO3','MT-CO1','CTSA','MT-ND4L','APOA1','MEP1A','AMN','NEAT1','MFI2','SLC5A12','MST1','IGF2','SLC15A1','ARID3A','RBP2','SLC6A19','MAMDC4','COL6A1','MSLN','SLC7A7','FABP1','ACTN4','ATP1A1','MYO15B','MT-ND2','SERPINA1','GNA11','PCSK5','SAT2','APOE','SEPP1','ALDOB','SLC26A3','APLP2','PHGR1','OAF','ANPEP','G0S2','NDRG1','VTN','GNS','PRODH','DST','SLC6A8','HEPH','TPP1','TPT1','ACSL5','APOC3','PTPRF','TFEC','AHNAK','MXRA8','VIL1','MYO1A','GDA','AOC1','KCNQ1OT1','C1orf115','EZR','CIB2','BTNL3','NR2F6','RARRES1','PBLD','HIF3A','FXYD3','S100A14','UGT2A3','FGFR3','MUC17','CREB3L3','FABP2','XPNPEP2','FBXO2','LAMB3','LPP','SLC39A5']
	
	sc.tl.score_genes(adata, clust_1_genes, ctrl_size=len(clust_1_genes), gene_pool=None, n_bins=25, score_name='clust_1', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, clust_2_genes, ctrl_size=len(clust_2_genes), gene_pool=None, n_bins=25, score_name='clust_2', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, clust_3_genes, ctrl_size=len(clust_3_genes), gene_pool=None, n_bins=25, score_name='clust_3', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, clust_4_genes, ctrl_size=len(clust_4_genes), gene_pool=None, n_bins=25, score_name='clust_4', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, clust_5_genes, ctrl_size=len(clust_5_genes), gene_pool=None, n_bins=25, score_name='clust_5', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, clust_6_genes, ctrl_size=len(clust_6_genes), gene_pool=None, n_bins=25, score_name='clust_6', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, clust_7_genes, ctrl_size=len(clust_7_genes), gene_pool=None, n_bins=25, score_name='clust_7', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, clust_8_genes, ctrl_size=len(clust_8_genes), gene_pool=None, n_bins=25, score_name='clust_8', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, clust_9_genes, ctrl_size=len(clust_9_genes), gene_pool=None, n_bins=25, score_name='clust_9', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, clust_10_genes, ctrl_size=len(clust_10_genes), gene_pool=None, n_bins=25, score_name='clust_10', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, clust_11_genes, ctrl_size=len(clust_11_genes), gene_pool=None, n_bins=25, score_name='clust_11', random_state=0, copy=False, use_raw=False)
	
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
	
	sc.pl.pca_loadings(adata, show=False, components=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], save='_PCA-loadings.pdf')
	
	## Draw the PCA elbow plot to determine which PCs to use
	sc.pl.pca_variance_ratio(adata, log=True, n_pcs = 100, save = '_elbowPlot.pdf', show = False)
	
	## Compute nearest-neighbors
	
	sc.pp.neighbors(adata, n_neighbors=num_neighbors_use, n_pcs=num_pcs_use)
	
	## fix batch differences based on XX/XY
	#bbknn.bbknn(adata, batch_key='sampleName', n_pcs=50, neighbors_within_batch=3, copy=False)
	
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
	sc.pl.umap(adata, color=['louvain', 'age'], save = '_clusterIdentity_age.pdf', show = False, legend_loc = 'right margin', edges = False, edges_color = 'lightgrey', edges_width = 0.01, size = 20, palette = greatestPalette, alpha = 0.95, legend_fontsize=6)
	sc.pl.umap(adata, color='age', save = '_age.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='tissue', save = '_tissue.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='sex', save = '_sex.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='gel', save = '_hydrogel.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='media', save = '_media.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = greatestPalette, alpha = 0.95)
	sc.pl.umap(adata, color='sampleName', save = '_sample.pdf', show = False, legend_loc = 'right margin', edges = False, size = 20, palette = greatestPalette, alpha = 0.95)
	
	## Run PAGA to get nicer cell starting positions for UMAP
	sc.tl.paga(adata, groups='louvain')
	sc.pl.paga(adata, color='louvain', save='_paga-init-pattern.pdf', show=False, threshold=threshold, node_size_scale=node_size_scale, node_size_power=0.9, layout=paga_layout)
	
	
	Enteroendocrine_Cell = ['CHGA','CPE','SCGN','NEUROD1','PCSK1N','TUBA1A','TM4SF4','SCG2','FEV','BEX1','CHGB','CRYBA2','PCSK1','CADPS','INSM1','SSR4','KCTD12','GC','SMIM6','NGFRAP1','NPDC1','RGS2','C9orf16','PTMS','SCG3','PAM','HEPACAM2','LY6H','HOPX','AES','EID1','SCG5','UCP2','BEX2','SEC11C','H3F3B','DDX5','GABARAPL2','GNAS','MAP1B','GAD2','KIAA1244','RAB3B','S100A6','QPCT','VWA5B2','PROX1','STMN2','NKX2-2','RAB26','MIR7-3HG','ALDH1A1','CBX6','RFX6','GTF2I','NLRP1','STMN1','RIMBP2','DNAJC12','PKM','TUSC3','DSP','MAP1LC3A','AK1','VAMP2','QDPR','C1QL1','CNIH2','LRRFIP1','CACNA1A','PPDPF','SCT','APLP1','NOVA1','TTR','SYP','MYL6','LINC00261','CACNA2D1','MARCKSL1','FXYD3','ABCC8','MARCKS','ZKSCAN1','PCBP4','KIAA1324','PFN2','TBCB','IDS','SYT13','SSTR2','GHRL','SNAP25','VTN','TUBA4A','MDK','EGR1','FOXA2','GCH1','C10orf10']
	Secretory_Cell = ['S100A6','LGALS4','GSN','SH3BGRL3','TFF3','HEPACAM2','HSPB1','MDK','LYZ','PPDPF','S100A11','FXYD3','SELM','FCGBP','TMSB4X','CLCA1','KRT19','RNASE1','GUCA2A','MYL6','HPCAL1','NEURL1','TPM1','SPINK1','TPD52','HES6','GUCA2B','SPIB','SERF2','ITLN1','FRZB','KRT8','LRRC26','MUC2','BCAS1','H3F3B','KLF4','PLP2','REP15','CD9','JUN','STARD10','ST6GALNAC1','YWHAZ','QSOX1','AGR3','MT2A','IER2','CALM2','GPX2','AGR2','KRT20','TMSB10','MGLL','FOS','KLK1','CDC42EP5','DLL1','BTG2','CTSE','HMGN2','TXNIP','WFDC2','MARCKSL1','LINC00261','FOXA3','CEACAM6','UCP2','ALDH2','XBP1','PRSS1','TCEA3','EGR1','ELF3','H3F3A','CA7','DYNLL1','RBPJ','RAB11FIP1','SPATS2L','TSPAN1','TSTA3','SLC38A2','C12orf57','GMDS','FOSB','RASSF6','SSR4','CEACAM5','DYRK4','WSB1','KIAA1324','SELK','ENHO','LPCAT4','BEST4','ABCA4','FOXP1','CFTR','SYTL2']
	Stem_cell = ['TSPAN8','HSP90AB1','GNB2L1','AGR2','SLC12A2','ELF3','IMPDH2','NPM1','JUN','PTMA','EEF1A1','PABPC1','HNRNPA1','BTF3','MYC','BEX1','TMSB10','ZFP36L2','ASCL2','EEF2','HMGCS2','NACA','NAP1L1','IER2','MARCKSL1','ZFAS1','TXN','HES1','CLU','EDN1','OLFM4','LGR5']
	Absorptive_Cell = ['APOC3','SEPP1','APOA4','APOA1','PRAP1','SERPINA1','B2M','CAPN3','FTH1','FABP2','APOB','SLC7A7','IL32','SLC40A1','ITM2B','MUC13','C19orf77','NEAT1','VAMP8','LGMN','SAT1','MAMDC4','AMN','CTSA','CDHR5','RHOC','SAT2','FAM3C','GRN','C8G','ID2','CTSH','GCHFR','TDP2','ASAH1','DAB2','PCSK1N','CLTB','MYL9','PHGR1','ATOX1','NEU1','FBP1','ETHE1','RTN4','ACSL5','CIDEB','NPC2','HLA-C','SLC39A4','FOLH1','LPGAT1','HLA-B','G0S2','CTSB','CD63','MAX','MFI2','KHK','GNS','DHRS7','PEPD','RARRES1','SLC5A12','SLC46A3','MME','AGPAT2','TMEM176A','SLC25A5','FN3K','FAM132A','ALDOB','NR0B2','CD74','HSD17B11','BTNL3','ALPI','CST3','REEP3','SLC15A1','ADA','RBP2','MAF','GPX4','TMEM92','ANPEP','SPG21','FBXO2','PLAC8','OCIAD2','TNFSF10','MXRA8','COX7A2L','SLC6A19','C19orf69','NR1H4','GLRX','RBP2','FTL','SLC7A7','FTH1','CTSA','AMN','MAMDC4','SLC26A3','RARRES1','S100A14','SERPINA1','APOA4','NPC2','FABP2','MYL9','C19orf77','MUC13','ALDOB','SEPP1','APOA1','APOC3','OAT','SAT2','GUK1','PRAP1','PEPD','B2M','ITM2B','C19orf33','ASS1','FCGRT','ASAH1','CDHR5','CD68','G0S2','FAM3C','SLC25A5','GCHFR','ID2','FBP1','SCT','FABP1','KHK','DHRS7','SMIM1','TTR','FOLH1','APOB','MS4A8','SLC5A12','CD63','PCSK1N','RHOC','ATP1B3','ESPN','ENPP7','MAF','COX7A2L','COX7C','ETHE1','TM4SF20','IL18','CIDEB','MTTP','IL32','SMLR1','FBXO2','F10','CBR1','SLC9A3R1','GRN','NR0B2','GOLT1A','CTSH','TDP2','ANPEP','SLC6A19','VAMP8','ANXA4','CST3','HSD17B11','PRR13','TMEM92','LGALS2','ACSL5','ADA','SLC40A1','PHPT1','SLC2A2','NEAT1','ATP5E','SCPEP1','SLC15A1','HADH','LGMN','TMEM176A','PLAC8','ATP6V1F']
	
	sc.tl.score_genes(adata, Enteroendocrine_Cell, ctrl_size=len(Enteroendocrine_Cell), gene_pool=None, n_bins=25, score_name='Enteroendocrine_Cell', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, Secretory_Cell, ctrl_size=len(Secretory_Cell), gene_pool=None, n_bins=25, score_name='Secretory_Cell', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, Stem_cell, ctrl_size=len(Stem_cell), gene_pool=None, n_bins=25, score_name='Stem_cell', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, Absorptive_Cell, ctrl_size=len(Absorptive_Cell), gene_pool=None, n_bins=25, score_name='Absorptive_Cell', random_state=0, copy=False, use_raw=False)
	
	scoreFigumap = plt.figure(dpi=80, figsize=(14,14))
	ax1 = scoreFigumap.add_subplot(2,2,1)
	ax2 = scoreFigumap.add_subplot(2,2,2)
	ax3 = scoreFigumap.add_subplot(2,2,3)
	ax4 = scoreFigumap.add_subplot(2,2,4)
	
	sc.pl.umap(adata, color='Enteroendocrine_Cell', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax1)
	sc.pl.umap(adata, color='Secretory_Cell', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax2)
	sc.pl.umap(adata, color='Stem_cell', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax3)
	sc.pl.umap(adata, color='Absorptive_Cell', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax4)
	
	ax1.set_title('Enteroendocrine_Cell')
	ax2.set_title('Secretory_Cell')
	ax3.set_title('Stem_cell')
	ax4.set_title('Absorptive_Cell')
	
	scoreFigumap.savefig(''.join([figure_dir,'/UMAP_InVivo_scoring_panels.pdf']))
	
	
	
	'''
	ageFigumap = plt.figure(dpi=80, figsize=(32,14))
	ax1 = ageFigumap.add_subplot(2,4,1)
	ax2 = ageFigumap.add_subplot(2,4,2)
	ax3 = ageFigumap.add_subplot(2,4,3)
	ax4 = ageFigumap.add_subplot(2,4,4)
	ax5 = ageFigumap.add_subplot(2,4,5)
	ax6 = ageFigumap.add_subplot(2,4,6)
	ax7 = ageFigumap.add_subplot(2,4,7)
	ax8 = ageFigumap.add_subplot(2,4,8)
	
	sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-0-NRG1-0'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax1)
	sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-0-NRG1-100'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax2)
	sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-1-NRG1-100'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax3)
	sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-10-NRG1-100'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax4)
	sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-100-NRG1-0'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax5)
	sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-100-NRG1-1'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax6)
	sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-100-NRG1-10'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax7)
	sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-100-NRG1-100'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax8)
	
	
	ax1.set_title('EGF-0ng/ml NRG1-0ng/ml')
	ax2.set_title('EGF-0ng/ml NRG1-100ng/ml')
	ax3.set_title('EGF-1ng/ml NRG1-100ng/ml')
	ax4.set_title('EGF-10ng/ml NRG1-100ng/ml')
	ax5.set_title('EGF-100ng/ml NRG1-0ng/ml')
	ax6.set_title('EGF-100ng/ml NRG1-1ng/ml')
	ax7.set_title('EGF-100ng/ml NRG1-10ng/ml')
	ax8.set_title('EGF-100ng/ml NRG1-100ng/ml')
	
	ageFigumap.savefig('UMAP_louvain_treatment_panels.png')
	'''
	
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
	
	mjc.write_marker_file(adata, file_out=''.join([figure_dir, '/marker_output.csv']))
	
	sc.tl.filter_rank_genes_groups(adata, groupby='louvain', use_raw=True, log=True, key_added='rank_genes_groups_filtered', min_in_group_fraction=0.5, min_fold_change=2, max_out_group_fraction=0.5)
	
	sc.pl.rank_genes_groups(adata, key='rank_genes_groups_filtered', n_genes=30, sharey=False, save = '_markerPlots.pdf', show = False)
	sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups_filtered',  n_genes=6, save = '_markerDotPlots.pdf', color_map=my_dot_cmap, show = False, mean_only_expressed=True, dot_min=0.2, dot_max=1, standard_scale='var')





print('\nDone with entire script execution')










