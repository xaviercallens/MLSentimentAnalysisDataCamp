rm -r trainsmall2
 
mkdir trainsmall2
mkdir trainsmall2/neg
mkdir trainsmall2/pos
 
cp 'find train/neg/ | head -1000' trainsmall2/neg
cp 'find train/pos/ | head -1000' trainsmall2/pos