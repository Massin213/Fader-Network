commande entrainement : python train.py   --data_dir ./Img/img_align_celeba   --attr_file ./Img/list_attr_celeba.txt   --eval_partition ./Img/list_eval_partition.txt   --batch_size 20   --epochs 10   --lr 1e-4   --latent_dim 128   --lambda_adv 0.003   --out_dir ./checkpoints3  --fraction 0.05

commande test: python display_inference.py \
  --checkpoint ./checkpoints3/fader_epoch_10.pth \
  --input_image ./Img/img_align_celeba/000001.jpg \
  --latent_dim 256 \
  --attr_index 31 \
  --attr_value 2.0


influence lambda:

0.01 : Assez faible, laisse beaucoup de place à la reconstruction.
0.1 : Plus fort, l’adversarial “pèse” plus lourd.
0.001 : On met très peu de pression pour cacher l’attribut (on se concentre presque uniquement sur la reconstruction).
Souvent, on commence à 0.01 ou 0.1 et on ajuste selon :

Les images reconstruites : Si la reconstruction est trop dégradée, réduisez lambda_adv.
L’effet de manipulation : Si la manipulation d’attribut est quasi invisible, augmentez lambda_adv.
La stabilité de l’entraînement : Si les pertes oscillent trop, vous pouvez ajuster petit à petit (par exemple un schedule progressif où on augmente lambda_adv au fil des epochs).