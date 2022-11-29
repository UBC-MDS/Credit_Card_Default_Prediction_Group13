# Makefile
# Author: Florencia D'Andrea
# Date: 2022-11-28

# all : doc/count_report.html 

# count the words
results/isles.dat : data/isles.txt src/countwords.py
	python src/countwords.py data/isles.txt results/isles.dat

results/abyss.dat : data/abyss.txt src/countwords.py
	python src/countwords.py data/abyss.txt results/abyss.dat

results/last.dat : data/last.txt src/countwords.py
	python src/countwords.py data/last.txt results/last.dat

results/sierra.dat : data/sierra.txt src/countwords.py
	python src/countwords.py data/sierra.txt results/sierra.dat

# create the plots
figures/isles.png : results/isles.dat src/plotcount.py
	python src/plotcount.py results/isles.dat figures/isles.png
    
figures/abyss.png : results/abyss.dat src/plotcount.py     
	python src/plotcount.py results/abyss.dat figures/abyss.png
  
figures/last.png : results/last.dat src/plotcount.py 
	python src/plotcount.py results/last.dat figures/last.png
  
figures/sierra.png : results/sierra.dat src/plotcount.py 
	python src/plotcount.py results/sierra.dat figures/sierra.png

# write the report
doc/count_report.html : figures/isles.png figures/abyss.png figures/sierra.png figures/last.png
	Rscript -e "rmarkdown::render('doc/count_report.Rmd')"
	

# clean :
#	rm -f results/*.dat
#	rm -f figures/*.png
#	rm -f doc/*.html
	
	
	