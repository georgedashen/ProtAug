# One level, all models, all methods
seed=42
PAM=PAM1_k10
Ident=Identity_k0.12
MLM=Esm2_MLM_k0.27
GPT=ProtGPT2_k0.13
level=10

CUDA_VISIBLE_DEVICES=0 nohup python fine-tuning-fix.py --model Esm2 --method noaug --batch 4 --gpu 0 --lr 1e-4 --seed $seed --output Esm2-fix-noaug > log/Esm2_noaug.log && \
	CUDA_VISIBLE_DEVICES=0 nohup python fine-tuning-fix.py --model DistilProtBert --method noaug --batch 4 --gpu 0 --lr 1e-4 --seed $seed --output DistilProtBert-fix-noaug > log/DistilProtBert_noaug.log && \
	CUDA_VISIBLE_DEVICES=0 nohup python fine-tuning-fix.py --model prot_bert_bfd --method noaug --batch 4 --gpu 0 --lr 1e-4 --seed $seed --output prot_bert_bfd-fix-noaug > log/bfd_noaug.log &

CUDA_VISIBLE_DEVICES=1 nohup python fine-tuning-fix.py --model Esm2 --method aug --augfile _$MLM --batch 4 --gpu 1 --lr 1e-4 --seed $seed --output Esm2-fix-$MLM > log/Esm2_Esm2_MLM_$level.log && \
	CUDA_VISIBLE_DEVICES=1 nohup python fine-tuning-fix.py --model DistilProtBert --method aug --augfile _$MLM --batch 4 --gpu 1 --lr 1e-4 --seed $seed --output DistilPortBert-fix-$MLM > log/DistilProtBert_Esm2_MLM_$level.log && \
	CUDA_VISIBLE_DEVICES=1 nohup python fine-tuning-fix.py --model prot_bert_bfd --method aug --augfile _$MLM --batch 4 --gpu 1 --lr 1e-4 --seed $seed --output prot_bert_bfd-fix-$MLM > log/bfd_Esm2_MLM_$level.log &

CUDA_VISIBLE_DEVICES=2 nohup python fine-tuning-fix.py --model Esm2 --method aug --augfile _$GPT --batch 4 --gpu 2 --lr 1e-4 --seed $seed --output Esm2-fix-$GPT > log/Esm2_ProtGPT2_$level.log && \
	CUDA_VISIBLE_DEVICES=2 nohup python fine-tuning-fix.py --model DistilProtBert --method aug --augfile _$GPT --batch 4 --gpu 2 --lr 1e-4 --seed $seed --output DistilProtBert-fix-$GPT > log/DistilProtBert_ProtGPT2_$level.log && \
	CUDA_VISIBLE_DEVICES=2 nohup python fine-tuning-fix.py --model prot_bert_bfd --method aug --augfile _$GPT --batch 4 --gpu 2 --lr 1e-4 --seed $seed --output prot_bert_bfd-fix-$GPT > log/bfd_ProtGPT2_$level.log &

CUDA_VISIBLE_DEVICES=3 nohup python fine-tuning-fix.py --model Esm2 --method duplicate --batch 4 --gpu 3 --lr 1e-4 --seed $seed --output Esm2-fix-duplicate > log/Esm2_duplicate.log && \
	CUDA_VISIBLE_DEVICES=3 nohup python fine-tuning-fix.py --model DistilProtBert --method duplicate --batch 4 --gpu 3 --lr 1e-4 --seed $seed --output DistilProtBert-fix-duplicate > log/DistilProtBert_duplicate.log && \
	CUDA_VISIBLE_DEVICES=3 nohup python fine-tuning-fix.py --model prot_bert_bfd --method duplicate --batch 4 --gpu 3 --lr 1e-4 --seed $seed --output prot_bert_bfd-fix-duplicate > log/bfd_duplicate.log &

CUDA_VISIBLE_DEVICES=4 nohup python fine-tuning-fix.py --model Esm2 --method aug --augfile _$Ident --batch 4 --gpu 4 --lr 1e-4 --seed $seed --output Esm2-fix-$Ident > log/Esm2_Identity_$level.log && \
	CUDA_VISIBLE_DEVICES=4 nohup python fine-tuning-fix.py --model DistilProtBert --method aug --augfile _$Ident --batch 4 --gpu 4 --lr 1e-4 --seed $seed --output DistilProtBert-fix-$Ident > log/DistilProtBert_Identity_$level.log && \
	CUDA_VISIBLE_DEVICES=4 nohup python fine-tuning-fix.py --model prot_bert_bfd --method aug --augfile _$Ident --batch 4 --gpu 4 --lr 1e-4 --seed $seed --output prot_bert_bfd-fix-$Ident > log/bfd_Identity_$level.log &

CUDA_VISIBLE_DEVICES=5 nohup python fine-tuning-fix.py --model Esm2 --method aug --augfile _$PAM --batch 4 --gpu 5 --lr 1e-4 --seed $seed --output Esm2-fix-$PAM > log/Esm2_PAM_$level.log && \
	CUDA_VISIBLE_DEVICES=5 nohup python fine-tuning-fix.py --model DistilProtBert --method aug --augfile _$PAM --batch 4 --gpu 5 --lr 1e-4 --seed $seed --output DistilProtBert-fix-$PAM > log/DistilProtBert_PAM_$level.log && \
	CUDA_VISIBLE_DEVICES=5 nohup python fine-tuning-fix.py --model prot_bert_bfd --method aug --augfile _$PAM --batch 4 --gpu 5 --lr 1e-4 --seed $seed --output prot_bert_bfd-fix-$PAM > log/bfd_PAM_$level.log &

wait
