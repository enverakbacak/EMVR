for f in "/home/ubuntu/Desktop/EMR/Article_Video/Video_Dataset_3/Frames/*/*.jpg"
do
     mogrify $f -resize 224x224! $f
done

