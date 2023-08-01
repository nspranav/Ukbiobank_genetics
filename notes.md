# Preprocessing

1) Downloaded eQTL file from [2.e](http://resource.psychencode.org/)

2) Mapped the snp_id to the rsid of ukbiobank using the file __'/data/analysis/collaboration/Multi-sites_genetics/GeneticDB/GenomeBuild/human_9606_b151_GRCh37p13/00-All_chrpos_rsid.txt'__

3) Filtered each imputed chromozome using command __plink --bfile ukb_imputed_imggen_chr* --extract ./QTL/rs_snps_eqtl.txt --make-bed --out ukb_cal_imggen_eQTL_matched__, files can be found at /data/users2/pnadigapusuresh1/Projects/ukbiobank/Data/QTL

4) Recoded each filtered chromozome using command __plink --bfile filtered_chr* --recode A --out filtered_chr+str(i)+_recoded'__ file can be found at /data/users2/pnadigapusuresh1/Projects/ukbiobank/Data/QTL

5) Selected total of __1105083__ SNPs out of __2120751__ eQTLs
6) Performed further pruning of the data using __plink --bfile filtered_chr* --indep-pairwise 500kb 1 0.2 --make-bed --out filtered_chr*_pruned__ file can be found at /data/users2/pnadigapusuresh1/Projects/ukbiobank/Data/QTL
7) selected total of __45,455__ from __1,105,083__
8) Further, slected genes that are related to alzheimers
9) 