import polars as pl
movies = pl.read_csv('ml-20m//movies.csv',has_header=True,separator=',',encoding='utf-8')
movies_long = pl.read_csv('ml-20m//movies_long.csv',has_header=True,separator=',',encoding='utf-8').unique()

movie_lens_format=movies.select(title=pl.col('title').
                           str.to_lowercase().
                           str.replace_all(r'[\W_]+', '').
                           str.replace_all(r'[\(\d+\)]', '').
                           str.replace_all(' ',''),
                     year=pl.col('title').str.extract(r'\(\d+\)', 0).
                           str.replace_all(r'[\W_]+', ''),
                     movie_id_lens=pl.col('movieId')).filter(pl.col('title')!='')

check=movie_lens_format.to_pandas()

movie_kaggle_format=movies_long.select(pl.col('title').
                   str.to_lowercase().str.replace_all(r'[\W_]+', '').
                   str.replace_all(' ',''),
                        year=pl.col('release_date').str.slice(0,4),
                         movie_id_kaggle=pl.col('id'))

merged=movie_lens_format.join(movie_kaggle_format,on=['title']).unique().with_columns(checks=1).\
    filter(abs(pl.col('year').cast(pl.Int64, strict=False)-pl.col('year_right').cast(pl.Int64, strict=False))<=0).\
    with_columns(ranks=pl.col('movie_id_kaggle').rank().over(pl.col('movie_id_lens')),
                 ranks2=pl.col('movie_id_lens').rank().over(pl.col('movie_id_kaggle'))).\
    filter((pl.col('ranks')==1) & (pl.col('ranks2')==1) ).\
    with_columns(new_id=pl.col('title').rank().cast(int))

movie_lens_merged=movies.join(merged.select('movie_id_lens','new_id'),how='inner',left_on='movieId',right_on='movie_id_lens')
movie_kaggle_merged = movies_long.join(merged.select('movie_id_kaggle','new_id'),how='inner',left_on = 'id',right_on='movie_id_kaggle')

movie_lens_merged.write_csv('data/movie_lens_merged.csv')