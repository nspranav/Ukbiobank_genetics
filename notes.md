# Preprocessing
1) MAF using __'--bfile /data/qneuromark/Data/UKBiobank/Data_genetic/ukb_imputed_imggen_chr7
  --geno 0.05
  --hwe 1e-06
  --maf 0.05
  --make-bed
  --out /data/users3/pnadigapusuresh1/Projects/ukbiobank/Data/chr7_maf'__

2) Downloaded eQTL file from [2.e](http://resource.psychencode.org/)

3) Mapped the snp_id to the rsid of ukbiobank using the file __'/data/analysis/collaboration/Multi-sites_genetics/GeneticDB/GenomeBuild/human_9606_b151_GRCh37p13/00-All_chrpos_rsid.txt'__

4) Filtered each imputed chromozome using command __plink --bfile ukb_imputed_imggen_chr* --extract ./QTL/rs_snps_eqtl.txt --make-bed --out ukb_cal_imggen_eQTL_matched__, files can be found at /data/users3/pnadigapusuresh1/Projects/ukbiobank/Data/QTL

5) Recoded each filtered chromozome using command __plink --bfile filtered_chr* --recode A --out filtered_chr+str(i)+_recoded'__ file can be found at /data/users3/pnadigapusuresh1/Projects/ukbiobank/Data/QTL

6) Selected total of __1105083__ SNPs out of __2120751__ eQTLs
7) Performed further pruning of the data using __plink --bfile filtered_chr* --indep-pairwise 500kb 1 0.2 --make-bed --out filtered_chr*_pruned__ file can be found at /data/users3/pnadigapusuresh1/Projects/ukbiobank/Data/QTL
9) selected total of __45,455__ from __1,105,083__
10) Further, slected genes that are related to alzheimers
11) 

12) Best model is Data driven with 5 layers [3771749-42.txt](/DataDriven/DataDriven_5_3771749-42.txt) epoch 40


# Results

1) Best imaging model is from job 
2) best Imaging genetic model is 
3) Using contrastive learning we found L1+L2+L4 gives best result