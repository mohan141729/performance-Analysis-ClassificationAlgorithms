window.onload = function() {
    let table = document.getElementById("metrics-table");
    let rows = table.rows;

    let maxValues = { accuracy: -Infinity, precision: -Infinity, recall: -Infinity, f1Score: -Infinity, auc: -Infinity };
    let highestMetricsCounts = Array(rows.length).fill(0); // To count how many max values are in each row

    // First, find the highest value in each metric column (Accuracy, Precision, Recall, F1 Score, AUC)
    for (let i = 1; i < rows.length; i++) {
        let cells = rows[i].cells;

        let accuracy = parseFloat(cells[1].innerText);
        let precision = parseFloat(cells[2].innerText);
        let recall = parseFloat(cells[3].innerText);
        let f1Score = parseFloat(cells[4].innerText);
        let auc = cells[5].innerText === "N/A" ? -Infinity : parseFloat(cells[5].innerText);

        if (accuracy > maxValues.accuracy) {
            maxValues.accuracy = accuracy;
        }
        if (precision > maxValues.precision) {
            maxValues.precision = precision;
        }
        if (recall > maxValues.recall) {
            maxValues.recall = recall;
        }
        if (f1Score > maxValues.f1Score) {
            maxValues.f1Score = f1Score;
        }
        if (auc > maxValues.auc) {
            maxValues.auc = auc;
        }
    }

    // Now, count how many of the values in each row are the highest in their respective columns
    for (let i = 1; i < rows.length; i++) {
        let cells = rows[i].cells;

        let accuracy = parseFloat(cells[1].innerText);
        let precision = parseFloat(cells[2].innerText);
        let recall = parseFloat(cells[3].innerText);
        let f1Score = parseFloat(cells[4].innerText);
        let auc = cells[5].innerText === "N/A" ? -Infinity : parseFloat(cells[5].innerText);

        let count = 0;
        if (accuracy === maxValues.accuracy) count++;
        if (precision === maxValues.precision) count++;
        if (recall === maxValues.recall) count++;
        if (f1Score === maxValues.f1Score) count++;
        if (auc === maxValues.auc) count++;

        highestMetricsCounts[i] = count; // Store the count of max values for this row
    }

    // Find the row with the highest number of maximum values
    let rowWithMostMaxValues = highestMetricsCounts.indexOf(Math.max(...highestMetricsCounts));

    // Highlight the row with the most highest values
    let row = rows[rowWithMostMaxValues];
    row.style.fontWeight = "bold";
    row.style.color = "black"; // black text color
    row.style.borderColor = "darkgreen"; // dark green border color
    row.style.borderWidth = "3px"; // 3px border width
    row.style.fontsize = "28px"; // 20px font size
    row.style.backgroundColor = "lightgreen"; // Light green background
};