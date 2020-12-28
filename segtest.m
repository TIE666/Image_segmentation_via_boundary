img = im2double(imread("test_images/zebra.jpg"));
size = [321,481];

figure();
subplot(1,2,1);imshow(imresize(img,size));
subplot(1,2,2);imshow(segment_image(imresize(img,size)));