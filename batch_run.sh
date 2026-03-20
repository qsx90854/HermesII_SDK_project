#!/bin/bash
# batch_run.sh - Linux automated batch runner for multiple datasets

# Configuration
# Configuration
EXECUTABLE="./SDK_DemoCode_v2_save_fall_result_pc"
LOG_DIR="batch_logs"

# Select Datasets to Run
# Options: "all" or space-separated list e.g., "data1 data2 data6"
DATASETS=${DATASETS:-"data13"} # "data8 data9 data10 data11 data12 data13 data14 data15 data16 data17" "data1 data2 data3 data4 data5 data6 data7"
echo "DEBUG: DATASETS is '$DATASETS'" 

# Ensure executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Please run 'make -f makefile2' first."
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

if [ "$DATASETS" == "all" ]; then
    INI_FILES=(app_config_data*.ini)
    echo "Starting batch run for ALL datasets..."
else
    INI_FILES=()
    for d in $DATASETS; do
        if [ -f "app_config_${d}.ini" ]; then
            INI_FILES+=("app_config_${d}.ini")
        else
            echo "Warning: app_config_${d}.ini not found, skipping."
        fi
    done
    echo "Starting batch run for selected datasets: $DATASETS"
fi

echo "Results will be saved in the '$LOG_DIR' directory."
echo ""

# Initialize Report
REPORT_FILE="batch_summary_report.txt"
echo "Batch Run Report - $(date)" > "$REPORT_FILE"
echo "--------------------------------------------------" >> "$REPORT_FILE"

# Loop through specific ini files for Debugging
for ini in "${INI_FILES[@]}"; do
    # Extract identifier (e.g., data1, data2) from the filename
    data_id=$(echo "$ini" | sed 's/app_config_\(.*\)\.ini/\1/')
    log_file="$LOG_DIR/output_${data_id}.log"
    interval_file_source=$(grep "Demo_Output_Dir" "$ini" | cut -d'=' -f2 | tr -d '\r')
    
    echo "[$(date +'%H:%M:%S')] Running $data_id ($ini)..."
    
    # 1. Prepare environment: copy to default app_config.ini
    cp "$ini" app_config.ini
    
    # 2. Execute SDK demo and capture logs
    rm -f detection_trace.txt
    $EXECUTABLE > "$log_file" 2>&1
    
    # Capture Trace
    if [ -f detection_trace.txt ]; then
        trace_file="$LOG_DIR/trace_${data_id}.txt"
        mv -f detection_trace.txt "$trace_file"
        echo "   -> Trace: $trace_file"
    else
        echo "   -> Trace: None (No Case5 triggers)"
    fi
    
    # 3. Check result (New Logic: Read Verification Report)
    tp=0
    fp=0
    fn=0
    
    report_file="$interval_file_source/verification_report.txt"
    if [ -f "$report_file" ]; then
        tp=$(grep "TP=" "$report_file" | cut -d'=' -f2 | tr -d '\r')
        fp=$(grep "FP=" "$report_file" | cut -d'=' -f2 | tr -d '\r')
        fn=$(grep "FN=" "$report_file" | cut -d'=' -f2 | tr -d '\r')
    elif [ -f "$interval_file_source/fall_count.txt" ]; then
        # Fallback if report not generated (e.g. executable crash or old version)
        count=$(cat "$interval_file_source/fall_count.txt")
        fp=$count # Assume all are unverified detections (FP-risk) unknown
        echo "   -> Warning: verification_report.txt not found. Using fallback count: $count"
    fi
    
    # 4. Determine Status
    status="PASS"
    detail="TP=$tp FP=$fp FN=$fn"
    
    if [ "$fn" -gt 0 ]; then
        status="FAIL"
        detail="Missed Fall (FN=$fn)"
    elif [ "$fp" -gt 0 ]; then
        status="FAIL" 
        detail="False Alarm (FP=$fp)"
    fi
    
    # Dataset Specific Overrides
    case $data_id in
        "data4"|"data5")
            # Negative: Ignore FN
            if [ "$tp" -eq 0 ] && [ "$fp" -eq 0 ]; then
                status="PASS"
                detail="Correctly Rejected (TP=0 FP=0)"
            else
                status="FAIL"
                 detail="False Alarm (TP=$tp FP=$fp)"
            fi
            ;;
        *)
            # Positive: Strict (Data 1/2/3/6/7)
            if [ "$fn" -gt 0 ] || [ "$fp" -gt 0 ]; then
                status="FAIL"
            fi
            ;;
    esac
    
    # Special Case: If TP=0 and FN=0 (e.g. Negative Dataset 1, 4, 5)
    # Then FP>0 is FAIL, FP=0 is PASS. (Handled above)

    echo "   -> $status: $detail"
    echo "   -> Log: $log_file"
    
    # Append to Report
    printf "%-10s | %-4s | %s\n" "$data_id" "$status" "$detail" >> "$REPORT_FILE"

done

echo "" | tee -a "$REPORT_FILE"
echo "========================================" | tee -a "$REPORT_FILE"
echo "Batch processing complete." | tee -a "$REPORT_FILE"
echo "Check '$REPORT_FILE' for summary."
echo "Check the '$LOG_DIR' folder for detailed logs."
echo "========================================" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"
echo "Summary Table:" | tee -a "$REPORT_FILE"
echo "dataset | TP | FP | FN | Status" | tee -a "$REPORT_FILE"
echo "--------|----|----|----|-------" | tee -a "$REPORT_FILE"

for ini in "${INI_FILES[@]}"; do
    data_id=$(echo "$ini" | sed 's/app_config_\(.*\)\.ini/\1/')
    interval_file_source=$(grep "Demo_Output_Dir" "$ini" | cut -d'=' -f2 | tr -d '\r')
    
    tp="-"; fp="-"; fn="-"; status="UNKNOWN"
    
    report_file="$interval_file_source/verification_report.txt"
    if [ -f "$report_file" ]; then
        tp=$(grep "TP=" "$report_file" | cut -d'=' -f2 | tr -d '\r')
        fp=$(grep "FP=" "$report_file" | cut -d'=' -f2 | tr -d '\r')
        fn=$(grep "FN=" "$report_file" | cut -d'=' -f2 | tr -d '\r')
        
        # Determine Status
        status="PASS"
        if [[ "$data_id" == "data4" ]] || [[ "$data_id" == "data5" ]]; then
             # Negative Logic: Expect NO falls. Fail if TP>0 or FP>0
             if [ "$tp" -gt 0 ] || [ "$fp" -gt 0 ]; then status="FAIL"; fi
        else
             # Positive Logic: Expect falls. Fail if FN>0 or FP>0
             if [ "$fn" -gt 0 ] || [ "$fp" -gt 0 ]; then status="FAIL"; fi
        fi
    fi

    printf "%-7s | %-3s | %-3s | %-3s | %s\n" "$data_id" "$tp" "$fp" "$fn" "$status" | tee -a "$REPORT_FILE"
done
