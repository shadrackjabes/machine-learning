A=imread('./pic_grayscale.jpg');
A1=imread('./pic_grayscale1.jpg');
A2=imread('./pic_grayscale2.jpg');
A3=imread('./pic_grayscale3.jpg');
%image(A);
%image(A1);
%image(A2);
%image(A3);
newA1=zeros(size (A));
newA2=zeros(size (A));
newA3=zeros(size (A));
newA1(1: size(A1, 1), 1: size(A1, 2)) = A1;
newA2(1: size(A2, 1), 1: size(A2, 2)) = A2;
newA3(1: size(A3, 1), 1: size(A3, 2)) = A3;
B = reshape(A,[786432],1);
B1 = reshape(newA1,[786432],1);
B2 = reshape(newA2,[786432],1);
B3 = reshape(newA3,[786432],1);
AA = B;
[m,n]=size(AA);
AA=[AA,ones(m,n).*B1];
AA=[AA,ones(m,n).*B2];
AA=[AA,ones(m,n).*B3];


