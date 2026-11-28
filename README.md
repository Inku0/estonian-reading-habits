# Analyzing Estonians' reading habits

## How to run
I recommend using `uv sync` to first install the dependencies, then use your preferred Jupyter notebook environment 
to run the code cells in `main.ipynb`.

Due to a recent major update to `polars`, `typol` does not work without a tiny patch, which is included in the file named `fix-typol.patch`.
To apply this patch, use `patch -p0 < fix-typol.patch`. The patch modifies `frame.py` under `.venv/lib/python3.14/site-packages/typol/` to remove a deprecated import and method.

NB! The patch is intended for Python 3.14 and will fail for other versions, though it can easily be updated.

## Dataset information
### Dataset name
Eestlane loeb: raamatuaasta lugemisuuring 2025

### Dataset description
General trends and changes in Estonians' book consumption and reading interests, to examine the diversity of reading interests and preferences, and to explain the social and cultural factors that promote or diminish interest in books.

### Dataset methodology
Web-based questionnaire, focus group interviews

### Files in the dataset
- Lugeja25.tab
  - Tab-separated results from the study, each line is a person's answers.
- Datamap_Lugeja25.xlsm
  - Explanations about the variables and how to read the results

### Dataset authors
- Marge Allandi - Eesti Rahvusraamatukogu
- Jaanika Anderson - Tartu Ülikooli raamatukogu
- Kaisa Esko - Kantar Emor
- Maarja Hollo - Eesti Kirjandusmuuseum
- Ragne Kõuts-Klemm - Tartu Ülikooli ühiskonnateaduste instituut
- Margus Küppar - Eesti Rahvusraamatukogu
- Marin Laak - Eesti Kirjandusmuuseum
- Marju Lauristin - Tartu Ülikooli ühiskonnateaduste instituut
- Liina Luhats-Ulman - Eesti Rahvusraamatukogu
- Virgo Siil - Tartu Ülikooli ühiskonnateaduste instituut
- Riina Tamm - Eesti Rahvusraamatukogu
- Kristel Veimann - Eesti Rahvusraamatukogu
- Peeter Vihalemm - Tartu Ülikooli ühiskonnateaduste instituut
- Aivar Voog - Kantar Emor

