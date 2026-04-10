# Cancer Genomics Analyzer

A web-based application for analyzing genomic data and assessing cancer risk using VCF (Variant Call Format) data.

## Project Structure

```
cancer-genomics-analyzer/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   ├── base.html         # Base template (navigation, styling)
│   ├── home.html         # Homepage
│   ├── analyze.html      # Analysis interface
│   └── 404.html          # 404 error page
├── static/
│   └── DNAimage.jpg      # Background image (optional)
└── uploads/              # Auto-created folder for file uploads
```

## Installation

### 1. Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### 2. Setup Steps

```bash
# Clone or navigate to your project directory
cd cancer-genomics-analyzer

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### 3. Access the Application
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## Features

### Home Page
- Overview of the application
- Quick start guide
- Feature highlights
- Tool description

### Analysis Page
- **File Upload:** Upload VCF format files
- **Text Input:** Paste VCF data directly
- **Analysis Options:**
  - High Impact Variants Only
  - Include Phenotype Data
  - Generate Detailed Report
  - Export Results

### Results Display
- Variant table with details:
  - Chromosome position
  - Gene name
  - Impact level (HIGH/MODERATE/LOW)
  - Risk score

## VCF Format Support

The application accepts VCF (Variant Call Format) v4.0+ files with the following columns:

| Column | Description |
|--------|-------------|
| CHROM  | Chromosome (1-22, X, Y, MT) |
| POS    | Position on chromosome |
| ID     | Variant ID (dbSNP ID or .) |
| REF    | Reference allele |
| ALT    | Alternate allele |
| QUAL   | Quality score |
| FILTER | Filter status (PASS or .) |
| INFO   | Additional information |

### Example VCF Data
```
##fileformat=VCFv4.2
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO
17      41196312        BRCA1_mut1      A       G       100     PASS    GENE=BRCA1
17      41246570        BRCA1_mut2      C       T       95      PASS    GENE=BRCA1
13      32889611        BRCA2_mut1      G       A       100     PASS    GENE=BRCA2
```

## Cancer Gene Database

The application recognizes these cancer-related genes:

- **HIGH Impact:** BRCA1, BRCA2, TP53, PTEN
- **MODERATE Impact:** CDH1, MLH1, PALB2, ATM
- **LOW Impact:** CHEK2 and others

## API Endpoints

### POST /api/analyze
Analyze genomic data

**Request:**
```json
{
  "vcf_data": "VCF format content",
  "vcf_file": "file upload (optional)",
  "high_impact_only": "true/false",
  "include_phenotype": "true/false",
  "detailed_report": "true/false",
  "export_results": "true/false"
}
```

**Response:**
```json
{
  "success": true,
  "analysis_id": "timestamp_id",
  "variants": [...],
  "report": {...},
  "total_variants": 3
}
```

### GET /api/results/{analysis_id}
Retrieve previous analysis results

### GET /api/export/{analysis_id}
Export analysis results as JSON

## Customization

### Change Background
Edit the `.dna-bg` section in `base.html`:
```css
.dna-bg {
    background-image: url("/static/your-image.jpg");
}
```

### Add New Genes
Edit the `cancer_genes` dictionary in `app.py`:
```python
cancer_genes = {
    'YOUR_GENE': {'impact': 'HIGH', 'risk': 0.90},
    ...
}
```

### Modify Risk Categories
Edit the `get_risk_category()` function in `app.py`

## Security Notes

- Maximum file upload size: 16MB
- File uploads are stored in the `uploads/` folder
- Always validate and sanitize user inputs in production
- Implement authentication for production use
- Use HTTPS in production environments

## Troubleshooting

### Issue: Image not loading
**Solution:** Ensure `DNAimage.jpg` is in the `static/` folder or update the path in `base.html`

### Issue: File upload not working
**Solution:** 
- Check `uploads/` folder permissions
- Verify file size is under 16MB
- Ensure correct file format (.vcf or .txt)

### Issue: Port 5000 already in use
**Solution:** 
```bash
python app.py --port 5001
# or change port in app.py:
app.run(port=5001)
```

## Disclaimer

This tool is for **research and educational purposes only**. It should not be used for clinical decision-making without consultation from qualified healthcare professionals. Always consult with a certified genetic counselor or oncologist for medical advice.

## License

Open source - Free to modify and distribute

## Support

For issues or questions, please review the code comments and VCF format guide in the application.

---

**Version:** 1.0.0  
**Last Updated:** 2026