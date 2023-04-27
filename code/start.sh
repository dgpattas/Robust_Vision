DATE=`date +%Y%m%d%H%M%S`
INPUT=0
mkdir 'data'
mkdir 'data/'$DATE'_clothing'

# clothing classes
#"__background__", 1"Shorts",2"Dress",3"Swimwear",4"Brassiere",5"Tiara",6"Shirt",7"Coat",8"Suit",
#               9"Hat","Cowboy hat","Fedora","Sombrero","Sun hat","Scarf","Skirt","Miniskirt","Jacket",
#               "Glove","Baseball glove","Belt","Necklace","Sock","Earrings","Tie","Watch","Umbrella",
#               "Crown","Swim cap","Trousers","Jeans","Footwear","Roller skates","Boot","High heels",
#               "Sandal","Sports uniform","Luggage and bags","Backpack","Suitcase","Briefcase",
#               "Handbag","Helmet","Bicycle helmet","Football helmet"

python read_and_detect_deep_learning.py --input $INPUT --threshold 0.5 --output 'data/'$DATE'_clothing' --classes 1 2 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
