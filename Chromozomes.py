#%%
import subprocess
import os


path = '/data/users3/pnadigapusuresh1/Projects/ukbiobank/Data'
os.chdir(path+'/QTL')
imputed_chr_path = '/data/qneuromark/Data/UKBiobank/Data_genetic'
prefix = 'ukb_imputed_imggen_chr'
postfix = '_maf'
eqTL_snp_filename = 'rs_snps_eqtl.txt'
output_file = 'filtered_chr'

# extract snps
# plink --bfile ukb_cal_imggen_3 --extract ./QTL/rs_snps_eqtl.txt --make-bed --out ukb_cal_imggen_eQTL_matched
#%%
for i in range(1,23):
    subprocess.run(['plink', '--bfile', imputed_chr_path+'/ukb_imputed_imggen_chr'+str(i),'--geno', str(0.05), '--hwe', str(1e-06), '--maf', str(0.05),'--make-bed', '--out', path+ '/chr'+str(i)+'_maf'])
    # subprocess.run(['plink','--bfile', path+'/chr'+str(i)+'_maf','--indep-pairwise','500kb','1','0.4','--make-bed','--out','filtered_chr'+str(i)+'_pruned'])
    # subprocess.run(['plink','--bfile', 'filtered_chr'+str(i) ,'--extract','filtered_chr'+str(i)+'_pruned.prune.in','--make-bed', '--out','filtered_chr'+str(i)+'_pruned'])
    # subprocess.run(['plink','--bfile', 'filtered_chr'+str(i)+'_pruned','--recode','A','--out','filtered_chr'+str(i)+'_pruned_recoded'])
#%%

