import pandas as pd
df = pd.read_csv(r"C:\DKFZ\HeiPorSPECTRAL_example\intermediates\tables\HeiPorSPECTRAL@median_spectra@polygon#annotator1.csv")
print(df.columns.tolist())
print(df.head(2).T)
