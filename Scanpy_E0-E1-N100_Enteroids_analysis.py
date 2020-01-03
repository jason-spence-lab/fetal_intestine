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


figure_dir = './figures/E0-E1-N100'
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
sample_list = ['150-2','150-3']


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

epi_cell_type_genes = ['EPCAM','CDX2','LGR5', 'OLFM4', 'CHGA', 'LYZ', 'MUC2', 'MUC5', 'VIL1', 'DPP4', 'FABP1', 'ALPI', 'SI', 'MGAM', 'SPDEF', 'SHH', 'IHH', 'PDGFA', 'SOX9','AXIN2','LGR5','ASCL2','CMYC','CD44','CYCLIND','TCF1','LEF1','PPARD','CJUN','MMP7','MSI1','LRP5','LRP6','BHLHA15','DEFA6','DEFA5','DEFA4','REG3A','REG3B','REG3G']

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


## Pasring function to get marker gene data similar to Seurat's marker output
def write_marker_file(adata, file_out='markers_output.csv', n_genes=50):
	print('Parsing markers...')
	marker_dict = adata.uns['rank_genes_groups']
	unique_list = [] 
	for x in adata.obs['louvain'].values.tolist(): 
		if x not in unique_list: 
			unique_list.append(str(x))
	
	outFile = open(file_out, 'w')
	outFile.write('logFC,gene,score,pval,padj,cluster\n')
	
	#i = 0
	
	parsed_dict = dict()
	
	for item in marker_dict:
		if type(marker_dict[item]) is not dict:
			cluster_data = []
			for subitem in marker_dict[item]:
				cluster_data.append(subitem.tolist())
			
			if str(item) not in parsed_dict:
				parsed_dict[str(item)] = cluster_data
	for cluster in range(0, len(unique_list)):
		for i in range(0, n_genes):
			line_out = []
			for marker_value in marker_dict:
				if type(marker_dict[marker_value]) is not dict:
					line_out.append(str(marker_dict[marker_value][i].tolist()[cluster]))
			line_out.append(str(cluster))
			outFile.write(','.join(line_out) + '\n')
	
	print('Saving marker data to:', file_out)
	outFile.close()



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
	
	Enteroendocrine = ['GFRA3','CHGB','TP53I11','NEUROD1','VWA5B2','CCK','RFX6','PRNP','PCSK1','SYT13','RPH3AL','FABP5','PAM','SCGN','APLP1','FEV','SCG5','CELF3','RESP18','NEUROG3','MAGED1','SCG3','PAX4','OLFM1','CPLX2','ISL1','GPX3','ANXA6','GNG4','MREG','MAP1B','BEX1','BAIAP3','DISP2','','LRP11','RIMBP2','SNAP25','KLHDC8B','FOXA2','GCK','PCSK1N','GDAP1L1','MAP3K15','KCNH6','PRODH2','LHFPL2','FAM183A','NKX2-2','PAX6','ADPRM','MYT1','KCNK16','TAC1','SCARB1','ACADSB','VIM','XPNPEP2','AC034228.4','PARP6','PLXNB1','CNOT6L','NCALD','SCG2','PHLDB2','PEG3','MAPRE3','AC244197.3','AMIGO2','DNER','SYP','TOX3','INSM1','ADORA3','TMEM106C','SSTR1','CBFA2T2','SLC39A2','RASD1','CACNA2D1','RAB36','AKNA','GHRL','C8orf37','RBFOX2','PDE1C','MAPK8IP2','SCN3A','SSTR5','LYPD1','MARCKS','RIIAD1','TRIT1','PTPRU','APBB1','GALR3','RAPGEF4','SPHKAP','GOLIM4','CDK2AP1','AC092143.1','TMEM182','FAM135A','FAM43A','GOLGA7B','SLC26A4','CHD7','CERKL','CPLX1','GALR1','GPR119','FAM160A2','PCP4L1','EFCAB1','MAML3','AP3B2','TF','RAB31','HNRNPH3','FFAR1','EMB','TH','PTPRN','PRKAR1B','DOCK4','KIRREL2','SH2D5','TMEM130','PDE11A','NEK5','PNMA8A','RNF122','CHST11','TEKT2','TRPM2','MAP9','CTIF','BTBD17','RUFY2','AMBP','PKIA','MAPKBP1','UNC13A','GATM','SLC35D3','SPRED3','ZC3H12C','MAPK15','MARCH4','NEUROD2','CIDEA','KLHL32','HRH3','SLC8A1','KLHL31','GFRA1','ADGB','LHX1','PLK5']
	Enterocyte = ['MEP1B','ANPEP','APOA1','IGSF9','IL18','ACE2','CREB3L3','KRT20','SLC9A3','DPEP1','SLC25A45','RBP2','MS4A18','REG1B','CLEC2D','SLC51B','CYP2D6','BCO2','SLC3A1','CYP3A4','SLC16A5','MAOB','SI','ACAD11','EDN2','SULT2B1','SLC7A7','DGAT2','ENPEP','FMO5','FAM3B','SLC26A6','MPP1','APOA4','SLC5A11','C11orf86','ECI2','CYP4F2','ACE','HSD17B6','RDH16','ALPG','GPD1','PTPRH','PAPSS2','GGT1','ALDH1A1','NAALADL1','HSD17B2','EXOC3L4','HPGD','GNPDA1','MS4A10','UGT2A3','UPP1','LRRC19','FMO4','HKDC1','NR1H3','AGMO','SLC6A20','SOAT2','CES2','BCL2L15','ENTPD5','CNDP2','TMEM37','GDA','ABCG5','MOGAT2','ABHD3','ST3GAL4','SLC5A1','TUBAL3','GSTM1','SPHK1','SLC26A3','TMEM106A','SLC27A4','SOWAHA','SLC6A4','MME','ADAMTSL5','ALDH1L1','GPT','EMP1','COX7A1','APOC3','ABCG8','PEX11A','OSGIN1','SLC28A1','C17orf78','NAT8','NR1I3','SLC51A','FABP1','ABCC2','APOB','MICAL2','MGAT4C','HLA-A','HDHD3','SEC23A','SLC7A9','TMEM86A','NPC1L1','BTNL2','ACOT9','PAQR7','CBLC','TMEM253','SMLR1','ABHD6','AMN','PBLD','MTTP','AP2A2','PTK6','VWCE','CIDEB','SCO2','GRAMD2B','DPYD','ABAT','SLC46A1','ADTRP','XDH','TGFBI','CHP2','KHK','LCT','ATP6V0A2','RHBG','TMEM82','GALM','C1orf116','AC027796.3','SLC15A1','SULT1B1','SLC13A1','PM20D1','FAHD1','TRIM31','OPTN','MYO7A','SLC37A4','PPARGC1A','STOM','REEP6','CMBL','CDKN2B','PGM2','MAF','CTAGE1','SLC11A2','SPSB1','TMEM236','CD36','TREH','GSTK1','LIPE','TMEM139','GSDMD','OCM','SRXN1','LMBR1L','LPGAT1','FEZ2','SLC52A2','MOCOS','NEK3','TM6SF2','AGPAT2','SLC23A2','XKR9','TOB1','CLCN2','HECTD3','TBC1D22A','CTSS','SLC9A2','CDC42EP2','MALL','PLA2G12B','RHOD','KBTBD11','ACOX1','ARHGAP26','AC104389.5','TCN2','MYLK','THNSL2','DHRS1','ADH4','DGKQ','AC090227.2','DNM1','FRK','TSC22D3','SLC35F5','AC011479.1','S100G','UGDH','HAGH','XPNPEP1','COBL','HNF4G','PARP9','SGPL1','PCCB','SLC2A2','EPHX2','KCNK5','LRP1','TMEM135','DUSP12','ABCB1','TMEM252','SLC7A8','C11orf54','TM4SF5','AKR1B10','TMEM230','ACBD4','CRAT','PCSK5','GALT','GSTP1','ILVBL','URGCP','CHCHD7','CA4','SLC13A2','EPHA1','DAB1','SEPTIN9','ADIPOR2','CAST','CASP6','ITGA3','RILP','NKIRAS2','MARCH6','PLIN3','RAB11FIP3','RETSAT','ARG2','SLC39A5','PEPD','IDH1','CCDC134','MGAM','CEACAM20','SLC2A9','FRMD8','SMPDL3A','SLC5A9','GNA11','PLS1','RAB17','LGALS3','SLC25A37','GPR155','SPNS2','ACOT11','VMP1','MERTK','ZZEF1','BCHE','ABCD3','AQP11','GCNT2','ACSL5','GNG12','CDA','FCGRT','SEMA4G','ZFYVE21','PFKFB4','KIAA0319','CYP4V2','KIAA1211','PTDSS1','AC093525.2','CYB5B','MAOA','VAT1','EHHADH','SLC3A2','DHRS11','SH3TC1','IRAK2','STX12','DGAT1','ACAA1','CYP4F8','SNX9','AHNAK','EDN3','ZDHHC7','PPP1R14D','SLC43A2','FAAH','TYMP','ABHD14A-ACY1','CYB5R3','RNF13','RXRA','DQX1','SNX13','TICAM1','SIDT2','FAM78A','ALDH18A1','RMDN3','SAT1','CKMT1A','TXLNG','SLC31A1','SLC25A36','SLC25A34','KIAA0319L','MARC2','ALDOB','DECR1','SH3D21','UGT1A1','CCS','KIFC3','SLC18B1','APRT','SLC22A1','ACP6','OGDH','TFG','TSTD1','KLC4','ITPK1','BMP3','PLD1','EZR','CORO2A','CKB','FARP2','PXDC1','SAR1B','SCP2','GGACT','CST6','SFT2D2','ABR','SLC34A2','FAM160A1','PCYT1A','TEP1','HADHA','CCDC88C','LPCAT3','TBC1D14','GUCD1','ACADM','MVP','ACTN4','TSPAN15','RUFY3','MCU','SPINT1','SFXN1','ALAS1','TOR1AIP2','CARD16','BPNT1','BAIAP2L1','IFNGR2','AL139011.2','ATP1A1','DNPEP','AKR7L','DLST','UGT1A6','MYO1D','TMEM120A','CDH17','ACAA2','HADH','CASP7','ACP5','RFK','ALDH9A1','VIPR1','TXNDC17','ENO1','HSD17B4','SLC39A4','NLRP6','PTTG1IP','IL17RC','NET1','LAD1','MDH2','C9orf64','ERBB3','PROZ','TAX1BP3','PGD','GPI','PRAP1','LYPLA1']
	Enterocyte_progenitor = ['CCNB1','CDC20','CENPA','CDKN3','CDC25C','CCNB2','KIF22','UBE2C','SAPCD2','RBP7','CCNA2','AURKA','CDKN2D','KIF23','NEK2','BIRC5','PLK1','TACC3','MELK','CDCA3','HMMR','SPC25','TPX2','ARHGEF39','BUB1B','KIF4A','MAD2L1','AC074143.1','GPSM2','CKAP2L','KNSTRN','ID1','CMC2','CENPE','PIF1','CKAP5','CNIH4','SPC24']
	Goblet = ['ZG16','TFF3','AGR2','SCIN','PDIA5','TPSG1','CHST4','BCAS1','BACE2','GALNT12','REP15','S100A6','CAPN9','SPDEF','ATOH1','GUCA2A','PLA2G10','MLPH','SCNN1A','ERN2','TTC39A','LIPH','C1GALT1C1','KCNK6','CREB3L4','SLC12A8','PTPRR','KLK1','TNFAIP8','LRRC26','C1GALT1','GALNT7','FAM174B','SGSM3','GALNT3','SPATS2L','CCL15','SYTL2','CA8','UAP1','ASPH','SLC50A1','SMIM14','CREB3L1','HGFAC','STARD3NL','TSPAN13','GSN','CAPN8','GCNT3','TXNDC5','ATP2C2','HPD','BHLHE40','TFCP2L1','QSOX1','ST3GAL6','RAP1GAP','KCTD14','KDELR3','GALNT10','DNAJC10','SYTL4','HID1','SAMHD1','FKBP11','GALNT5','TMED3','ICA1','SLC66A3','TMEM123','SDF2L1','S100A14','ERGIC1','FOXA3','STX17','AC020907.6','CSRP1','PDIA6','TINAGL1','RCAN3','FAM114A1','CMTM7','MON1A','SLC7A4','TNFRSF21','TOR3A','P2RX4','MYO5C','NIPAL2','TMEM39A','SIL1','SLC17A9','MCF2L','AC093668.1','CGREF1','GALK2','WARS','EDEM1','SLC35A1','TM9SF3','FHL1','SEC24D','SEL1L3','TMED9','CD9','RASD2','EDEM2','GOLPH3L','ARFIP2','TSTA3','TVP23B','RNF39','AACS','CHRM1','FUT4','VPS37C','CRELD2','IKBIP','NANS','AC036214.3','TMEM214','ANXA3','RASSF6','BCAT2','TMEM159','STXBP6','SLC30A7','MANSC1','GFPT1','GMPPB','SYBU','SRD5A1','SLC39A7','TMEM248','BET1L','SEC23IP','COG6','RAB3D','C9orf152','PRRC1','APPL2','C11orf24','SYNJ2','SMIM33','ARFGAP1','FAM3D','EHD4','STX5','PLCB1','PTGER4','SLC39A11','PLLP','GPR20','SPINK4','NFKB2','TMCO3','MLLT3','GMPPA','CDK5RAP3','PARM1','KCNH3','TSPAN1','B3GNT7','ENTPD4','KDELR2','SPPL2A','IMPAD1','MGAT3','CPD','ASNS','HYOU1','UBA7','DNAJC3','GOLT1B','PYGB','MANF','XBP1','GALNTL6','HSPA13','RAB27B','RASEF','ITGA2','GORASP1','PCK1','PGM3','GALNT6','GOLGA5','SEC16A','EIF2AK3','OSBPL2','ZNF467','HDLBP','CBFA2T3','ZBP1','B3GNT5','FAR1','ZNF330','GCC2','LMAN1','LAMC2','HERPUD1','SLC10A7','SERP1','SCAMP1','GAL3ST2','ODF2L','HILPDA','COG3','ALYREF','GALNT4','LITAF','FAM98A','PCSK9','ZBTB8A','TMEM63A','TRIM47','SSR3','EDEM3','TST','ANG','SLC38A10','GUK1','PCSK7','TRABD','GFI1','GNPNAT1','PDXDC1','HSPA5','SLC35A2','ARL1','SMIM5','CCND3','SAR1A','F2RL1','STT3A','TDRD7','SIDT1','PDIA3','LSS','CMPK1','NAGA','SH3BGRL3','SLC41A2','OSTC','FGFR3','FUT8','GGCX','PLAC9','SEC61B','BSCL2','GOLM1','KLF4','SSR4','SRPRB','YIPF6','CLPTM1L','ID4','ARF4','GALE','EIF4EBP1','SRPRA','TBC1D30','ZC3H7A','S100A16','MKNK2','TMPRSS2','TC2N','SLC35C1','UFSP2','TMEM165','SEC62','CYP51A1','FAM3C','SLC49A3','SLC37A1','CMTM8','ADAM9','CAPNS1','SYT7','PDIA4','SLC22A23','YIPF5','ATF4','ICK','SRM','PLAUR','PYROXD1','FRY','CYP2J2','STK38L','SPRYD3','GNE','ALDH3B2','RELL1','KRTCAP2','SEC23B','ST3GAL1','TLCD4','TULP4','CAPN7','GPR180','TXNDC11','COPB2','CALR','HOMER2','SSR2','TBRG1','JTB','SYVN1','MORF4L2','RPN2','UGP2','HM13','SLC16A6','SLC39A1','AC136352.3','COPG1','SSR1','TMED2','ANK3','AC078927.1','RPN1','UGGT1','PPIB','CAMSAP3','DDOST','C19orf71','SH3BGRL2','GOLGB1','B3GNT3','DCBLD2','SPCS2','SEC61A1','CANT1','TPCN1','GORASP2','PMM2','ANO7','RRBP1','PACSIN1','TNK2','EIF2AK4','SEC22B','TARS','SLC1A5','COPB1','YIF1B','ETNK1','RAMP1','CLTB','SLC22A15','KIF13A','YIPF3','IFT20','UFL1','TM9SF2','SYNGR2','NUCB1','GMDS','SEC61G','RFC1','C2CD2L','SMIM3','HSP90B1','SRP9','OST4','TMEM183A','AC068631.2','TOM1L1','SH3PXD2A','AC012254.2']
	Paneth = ['DEFA1','LYZ','CLPS','REG4','PNLIPRP2','PLA2G2A','ITLN1','MMP7','ANG','HABP2','PNLIPRP1','SAMD5','C4BPA','APOC2','PLA2G2F','COPZ2','KLF15','SNTB1','GGH','FZD9','FGFRL1','TESC','SLC1A4','LAMB1','SLC30A2','HSPB8','SYNC','SLC16A7','HAPLN4','INSRR','ACVR1C','SYNE4','ACOX2','DKK3','THBS1','DLL3']
	Stem = ['LGR5','GKN3P','ASCL2','OLFM4','RGMB','IGFBP4','JUN','PDGFA','SOAT1','TNFRSF19','CYP2E1','FSTL1','IFITM1','PRELP','SCN2B','HLA-DQB1','SLC1A2','CD74','SP5','NOXA1','RGCC','SORBS2','SECTM1','CDO1','SLC14A1','CLCA2','TIFA','PLS3','HMGCS2','ARID5B','AGR3','SLC12A2','RASSF5','AC004687.2','NRN1','LAMB3','CD44','AXIN2','SLC27A2','AFAP1L1','CCDC3','LRIG1','NOXO1','CDK6','TGIF1','TNS3','NR2E3','EFNA4','RNF32','PRSS23','SMOC2','MECOM','ESRRG','AC004691.2','ZNRF3','GRB7','PHGDH','AQP4','LCP1','CA12','ZBTB38','CDCA7','FAM13A','SHISA2','DTX4','SLC19A2','CD14','MYO9A','APP','CLIC6','WEE1','LANCL1','CASP12','SH3RF1','LRP4','ARHGEF26','ETV6','CTTNBP2','SLC16A13','HTR4','PDXK','IMMP2L','NAP1L1','SDC4','EPN3','SIPA1L1','ZNF341','NGEF','NRG4','CSAD','RIN2','CD81','IRF2BP2','SESN3','PHLPP1','YAP1','MFGE8','ITGA1','PCDH8','VDR','KCNQ1','SLC28A2','ZFP36L1','UROD','RGS12','NFIB','SDSL','NFIA']
	Tuft = ['ALOX5AP','HCK','LRMP','AVIL','TRPM5','AC020909.1','RGS13','LTC4S','PYGL','SH2D7','DCLK1','ALOX5','PIK3R5','FYB1','VAV1','MATK','TSPAN6','STRIP2','POU2F3','C11orf53','PTPN6','BMX','TUBA1A','ESPN','PLCB2','FFAR3','PLCG2','LY6G6F','HPGDS','PEA15','LY6G6D','PIK3CG','INPP5D','CCDC28B','SNRNP25','KCTD12','SKAP2','GPRC5C','RGS22','GFI1B','HMX3','CBR3','PFKFB3','PRSS53','ITPR2','LIMD2','CD300LF','AC004593.3','SMPX','PTGS1','A4GALT','RAC2','CSK','SLCO4A1','PTPN18','CHAT','HEBP1','PPP1R14C','DGKI','INPP5J','TPPP3','GNG13','ILDR1','CWH43','IL17RB','NCF2','COPRS','DDAH1','TMEM116','SUCNR1','TMEM176A','ADCY5','FNBP1','PLK2','HMX2','TMEM141','KRT23','GPRC5A','RGS2','CAMK2B','FES','BPGM','ACACB','IL13RA1','ZNF428','PPP1R3B','CCNJ','BCL2L14','TMEM229A','ETHE1','RUNX1','GGA2','APOBEC1','SERPINI1','AL157935.2','INPP5B','SAMD14','PGM2L1','PLA2G4A','PTPRC','AC002996.1','PNPLA3','JARID2','RGS19','REEP5','TIPARP','GNAI2','FAM49A','CACNA2D2','YPEL2','ACOT7','SVIL','ABHD16A','TRIM40','TRAK1','SEC14L1','BNIP5','SMTN','GALK1','TBC1D1','TMEM176B','ABHD2','HSBP1L1','SLC4A8','MYO1B','TMEM38B','HK1','NEURL1','DMXL2','BUB3','PTPRJ','TRIB2','STARD5','UBTD1','SLC41A3','PLEKHG5','RBM38','TLCD3A','EEF2K','CABLES2','FBXO25','AP1S2','ERO1B','CLMN','FAM49B','CPVL','PRR15','LPCAT4','TMEM74B','MN1','EPPK1','SAMD9L','TMEM245','GLYCTK','ALDH3A2','PPP3CA','CPNE3','SLC4A7','NFATC1','KIT','FAM117B','TMEM121','CPM','ASAH1','SLC9A9','UBL7','ABCA3','PDE6D','BMP2','AL451062.3','CAMKK2','ARHGAP8','AGT','PTPRA','ADH1A','DUSP14','CLIC4','GIMAP1','CPNE5','CEACAM1','GCNT1','B4GALT5','SUCO','PIM3','OGDHL','OAS1','DCP1B','GCOM1','CDKN1A','CD37','BRMS1','LRRC42','PLD2','TMEM9','CPEB4','SSX2IP','DDAH2','TMEM65','C12orf43','MSI2','B4GALT4','RABGAP1L','AL358075.4','NT5C3A','PALLD','C15orf48','PIP5K1B','KRT18','MAP1A','LMF1','ARHGEF28','NSFL1C','TXNDC16','PSTPIP2','TTLL11','EXPH5','GADD45A','PLEKHS1','JMY','ATAT1','ARHGEF2','LMBR1','RHOC','CARD10','KCNJ16','ARHGAP4','ACSL4','RHOG','FAM221A','DYNLT1','C2','ZBTB41','SOCS1','ATP6AP1','FAM171A1','WNK2','KCND3','SLC27A1','ATXN1','RABGAP1','MYRFL','CROT','TM4SF4','UBE2J1','SORT1','LIMA1','MOV10','LCA5','MLIP','C9orf16','CKAP4','TOR4A','RMDN1','OAS2','DSP','SOX9','OSBPL3','KIF21B','TBCB','ARAP2','CASP3','ENC1','IL25','LMAN2L','ZMIZ1','NAV2','ATP2A3','GIMAP8','FOLR1','FN1','HSPA4L','SUFU','ATP8A1','VPS53','RGS14','PDCL','SHKBP1','PKP1','IL4R','DVL1','ZFHX3','ADAM22','GRAMD1C','TMEM45B','UNC5B','MICAL3','KCTD13','AK7','TCTA','NEK7','AC069368.1','MYO6','CHDH','OPN3','TLE3','TTLL10','STRADA','YPEL3','CMIP','CACHD1','PIGC','ATP6V1D','RDX','S100A11','SPA17','GIMAP1-GIMAP5','CYSTM1','ZDHHC17','LECT2','VDAC3','HSPB11','SLC16A2','ABHD5','RHBDF1','CBLB','NFE2L3','SEPTIN8','GPCPD1','PSD3','ANXA11','SLC25A12','EHF','AKR1B10','DAPP1','ESYT1','PPT1','CD47','MICAL1','GNA14','PACS2','LYN','RMND5A','ANKRD12','RIT1','CAMTA2','MOCS2','USP49','NRBP2','AP000295.1','EPHA4','ARL5A','RGL2','ST18','C3orf52','TEAD1','ENPP4','TMEM158','TNFAIP3','GYS1','HIVEP2','CAP1','SLC4A2','MAP4K4','DESI1','MAN2A1','CYP17A1','CYHR1','MORF4L1','STOX2','HIST3H2A','HDAC6','PROX1','LRCH4','SPIRE2','KLF6','RAB5B','ANXA4','RAB4B','IQSEC1','PDPK1','STK40','GDE1','MTMR11','CIB2','MARCH2','CAPG','NARF','MGST3','ANGEL1','BICD1','IFITM1','STX3','S100A1','OMD','C4orf19','ARPC5','AC002985.1','CDC42SE1','ABCC3','HSF2','PNPLA6','CCDC68','FRYL','LMTK2','TAS1R3','USPL1','AJUBA','KALRN','BASP1','PIP5KL1','SLC26A2','ATP2B2','SMUG1','WDFY2','TRIM38','ARF3','SCAND1','DPYSL2','NDUFAF3','SIK1','WDR7','SFXN3','KCNQ4','HSBP1','CALML4','ATF7IP','HAP1','KCTD15','PRCP','GMIP','CMTM3','MADD','AC073508.2','NSF','KLHL28','PPARG','EML3','PHLDA1','P2RX1','PDE9A','OTUD7B','TFPI2','RILPL2','KLF3','GYG1','ARMCX1','LZTS2','PLEK','VAMP8','STAT2','TMEM160','TMEM51','CDHR5','STK38','ATP13A2','NPTN','SIRT5','GABARAPL2','NUDT14','ALKBH7','SLC18A3','TTLL7','ACSS2','SIAE']
	MCell = ['TNFAIP2', 'CCL9', 'GP2', 'SCG5']
	
	sc.tl.score_genes(adata, Enteroendocrine, gene_pool=None, n_bins=25, score_name='Enteroendocrine', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, Enterocyte, gene_pool=None, n_bins=25, score_name='Enterocyte', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, Enterocyte_progenitor, gene_pool=None, n_bins=25, score_name='Enterocyte_progenitor', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, Goblet, gene_pool=None, n_bins=25, score_name='Goblet', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, Paneth, gene_pool=None, n_bins=25, score_name='Paneth', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, Stem, gene_pool=None, n_bins=25, score_name='Stem', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, Tuft, gene_pool=None, n_bins=25, score_name='Tuft', random_state=0, copy=False, use_raw=False)
	sc.tl.score_genes(adata, MCell, gene_pool=None, n_bins=25, score_name='MCell', random_state=0, copy=False, use_raw=False)
	
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
	bbknn.bbknn(adata, batch_key='sampleName', n_pcs=50, neighbors_within_batch=3, copy=False)
	
	## Calculate cell clusters via Louvain algorithm
	
	sc.tl.louvain(adata, resolution = louv_res)
	
	## Run UMAP Dim reduction
	
	sc.tl.umap(adata, min_dist=umap_min_dist, maxiter=maxiter, spread=umap_spread, gamma=umap_gamma)
	
	## Run tSNE algorithm
	
	sc.tl.tsne(adata, n_pcs=num_pcs_use)
	
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
	
	
	ageFigumap = plt.figure(dpi=80, figsize=(16,7))
	#ax1 = ageFigumap.add_subplot(2,4,1)
	ax2 = ageFigumap.add_subplot(1,2,1)
	ax3 = ageFigumap.add_subplot(1,2,2)
	#ax4 = ageFigumap.add_subplot(2,4,4)
	#ax5 = ageFigumap.add_subplot(2,2,1)
	#ax6 = ageFigumap.add_subplot(2,2,2)
	#ax7 = ageFigumap.add_subplot(2,2,3)
	#ax8 = ageFigumap.add_subplot(2,2,4)
	
	#sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-0-NRG1-0'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax1)
	sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-0-NRG1-100'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax2)
	sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-1-NRG1-100'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax3)
	#sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-10-NRG1-100'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax4)
	#sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-100-NRG1-0'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax5)
	#sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-100-NRG1-1'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax6)
	#sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-100-NRG1-10'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax7)
	#sc.pl.umap(adata[adata.obs['sampleName'].isin(['HT323_EGF-100-NRG1-100'])], color='louvain', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, palette = greatestPalette, alpha = 0.95, ax=ax8)
	
	
	#ax1.set_title('EGF-0ng/ml NRG1-0ng/ml')
	ax2.set_title('EGF-0ng/ml NRG1-100ng/ml')
	ax3.set_title('EGF-1ng/ml NRG1-100ng/ml')
	#ax4.set_title('EGF-10ng/ml NRG1-100ng/ml')
	#ax5.set_title('EGF-100ng/ml NRG1-0ng/ml')
	#ax6.set_title('EGF-100ng/ml NRG1-1ng/ml')
	#ax7.set_title('EGF-100ng/ml NRG1-10ng/ml')
	#ax8.set_title('EGF-100ng/ml NRG1-100ng/ml')
	
	ageFigumap.savefig(''.join([figure_dir,'/UMAP_louvain_treatment_panels.pdf']))
	
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
	
	
	umap_scoreFigumap = plt.figure(dpi=80, figsize=(28,14))
	ax1 = umap_scoreFigumap.add_subplot(2,4,1)
	ax2 = umap_scoreFigumap.add_subplot(2,4,2)
	ax3 = umap_scoreFigumap.add_subplot(2,4,3)
	ax4 = umap_scoreFigumap.add_subplot(2,4,4)
	ax5 = umap_scoreFigumap.add_subplot(2,4,5)
	ax6 = umap_scoreFigumap.add_subplot(2,4,6)
	ax7 = umap_scoreFigumap.add_subplot(2,4,7)
	ax8 = umap_scoreFigumap.add_subplot(2,4,8)
	
	sc.pl.umap(adata, color='Enteroendocrine', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax1)
	sc.pl.umap(adata, color='Enterocyte', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax2)
	sc.pl.umap(adata, color='Enterocyte_progenitor', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax3)
	sc.pl.umap(adata, color='Goblet', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax4)
	sc.pl.umap(adata, color='Paneth', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax5)
	sc.pl.umap(adata, color='Stem', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax6)
	sc.pl.umap(adata, color='Tuft', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax7)
	sc.pl.umap(adata, color='MCell', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax8)
	
	ax1.set_title('Enteroendocrine')
	ax2.set_title('Enterocyte')
	ax3.set_title('Enterocyte_progenitor')
	ax4.set_title('Goblet')
	ax5.set_title('Paneth')
	ax6.set_title('Stem')
	ax7.set_title('Tuft')
	ax8.set_title('M')
	
	umap_scoreFigumap.savefig(''.join([figure_dir,'/UMAP_InVivo_scoring_panels.pdf']))
	
	
	tSNE_scoreFigumap = plt.figure(dpi=80, figsize=(28,14))
	ax1 = tSNE_scoreFigumap.add_subplot(2,4,1)
	ax2 = tSNE_scoreFigumap.add_subplot(2,4,2)
	ax3 = tSNE_scoreFigumap.add_subplot(2,4,3)
	ax4 = tSNE_scoreFigumap.add_subplot(2,4,4)
	ax5 = tSNE_scoreFigumap.add_subplot(2,4,5)
	ax6 = tSNE_scoreFigumap.add_subplot(2,4,6)
	ax7 = tSNE_scoreFigumap.add_subplot(2,4,7)
	ax8 = tSNE_scoreFigumap.add_subplot(2,4,8)
	
	sc.pl.tsne(adata, color='Enteroendocrine', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax1)
	sc.pl.tsne(adata, color='Enterocyte', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax2)
	sc.pl.tsne(adata, color='Enterocyte_progenitor', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax3)
	sc.pl.tsne(adata, color='Goblet', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax4)
	sc.pl.tsne(adata, color='Paneth', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax5)
	sc.pl.tsne(adata, color='Stem', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax6)
	sc.pl.tsne(adata, color='Tuft', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax7)
	sc.pl.tsne(adata, color='MCell', save = False, show = False, legend_loc = 'on data', edges = False, size = 50, cmap = my_feature_cmap, alpha = 0.95, ax=ax8)
	
	ax1.set_title('Enteroendocrine')
	ax2.set_title('Enterocyte')
	ax3.set_title('Enterocyte_progenitor')
	ax4.set_title('Goblet')
	ax5.set_title('Paneth')
	ax6.set_title('Stem')
	ax7.set_title('Tuft')
	ax8.set_title('M')
	
	tSNE_scoreFigumap.savefig(''.join([figure_dir,'/tSNE_InVivo_scoring_panels.pdf']))
	
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
	
	mjc,write_marker_file(adata, file_out=''.join([figure_dir, '/marker_output.csv']))
	
	sc.tl.filter_rank_genes_groups(adata, groupby='louvain', use_raw=True, log=True, key_added='rank_genes_groups_filtered', min_in_group_fraction=0.5, min_fold_change=2, max_out_group_fraction=0.5)
	
	sc.pl.rank_genes_groups(adata, key='rank_genes_groups_filtered', n_genes=30, sharey=False, save = '_markerPlots.pdf', show = False)
	sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups_filtered',  n_genes=6, save = '_markerDotPlots.pdf', color_map=my_dot_cmap, show = False, mean_only_expressed=True, dot_min=0.2, dot_max=1, standard_scale='var')
	




print('\nDone with entire script execution')










