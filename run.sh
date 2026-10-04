#!/usr/bin/env bash
# End-to-end reproduction script for CSE-307 Term Paper

set -e

echo "=========================================================="
echo "CSE-307: Operating Systems - Term Paper Automation Suite"
echo "Track 1: Learned Page Replacement & Calibration Bonus"
echo "=========================================================="

echo -e "\n[1/3] Running Unit Test Suite..."
python3 -m unittest discover tests

echo -e "\n[2/3] Running Full Simulation, Shift Benchmarks, and Generating Figures..."
python3 experiments/run_experiments.py

echo -e "\n[3/3] Compiling Camera-Ready 3-Page PDF Report..."
python3 docs/generate_pdf_report.py

echo -e "\n=========================================================="
echo "ALL ARTIFACTS SUCCESSFULLY PRODUCED!"
echo "PDF Report : docs/CSE307_Term_Paper_Report.pdf"
echo "Charts     : results/*.png"
echo "Data Tables: results/*.csv, results/*.md"
echo "=========================================================="
