#%%
import subprocess
import os


path = '/data/users2/pnadigapusuresh1/Projects/ukbiobank/Data'
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
    #subprocess.run(['plink', '--bfile', path+'ukb_cal_imggen_5', '--chr' ,str(i),'--make-bed', '--out', path+ '/chr'+str(i)])
    #subprocess.run(['plink', '--bfile', 'Data/chr_'+str(i), '--geno',str(0.01),'--make-bed', '--out','Data/geno_001/chr_'+str(i)])
    subprocess.run(['plink','--bfile', 'filtered_chr'+str(i)+'_pruned','--recode','A','--out','filtered_chr'+str(i)+'_pruned_recoded'])
    #subprocess.run(['ls'])
    #subprocess.run(['plink','--bfile', 'filtered_chr'+str(i) ,'--extract','filtered_chr'+str(i)+'_pruned.prune.in','--make-bed', '--out','filtered_chr'+str(i)+'_pruned'])
    #subprocess.run(['plink','--bfile', 'filtered_chr'+str(i),'--indep-pairwise','500kb','1','0.1','--make-bed','--out','filtered_chr'+str(i)+'_pruned'])
#%%

