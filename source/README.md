
Launch as: 
python langdetect.py -i ../data/dataset.csv -v 1000 -a char
python langdetect.py -i ../data/dataset.csv -v 1000 -a word


Ideas:
- Get to know the dataset:
	- Find mostfrequent digrams -> This is features (but we don't know in what order)
	- Find distribution of digrams per language


TO DO:
	Apply if not Chinese or Japanese: 
		- Tokenization
		- Stop word removal
		- Lematization


