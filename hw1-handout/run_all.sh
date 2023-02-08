rm output/*.txt

python main.py --load_saved_matrix --algo GPR --scorer NS 
# python main.py --load_saved_matrix --algo GPR --scorer WS
# python main.py --load_saved_matrix --algo GPR --scorer CS


python main.py --load_saved_matrix --algo QTSPR --scorer NS --beta 0.15 --gamma 0.05
# python main.py --load_saved_matrix --algo QTSPR --scorer WS
# python main.py --load_saved_matrix --algo QTSPR --scorer CS


python main.py --load_saved_matrix --algo PTSPR --scorer NS --beta 0.15 --gamma 0.05
# python main.py --load_saved_matrix --algo PTSPR --scorer WS
# python main.py --load_saved_matrix --algo PTSPR --scorer CS