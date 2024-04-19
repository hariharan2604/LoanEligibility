let currentPage = 1;

function nextPage(page) {
    document.getElementById(`page${currentPage}`).style.display = "none";
    document.getElementById(`page${page}`).style.display = "block";
    currentPage = page;
}

function previousPage(page) {
    document.getElementById(`page${currentPage}`).style.display = "none";
    document.getElementById(`page${page}`).style.display = "block";
    currentPage = page;
}

// Updated submitForm function to show modal on success
function submitForm() {
    var formData = {
        Loan_ID: ["LP001015"],
        Gender: document.getElementById("gender").value,
        Married: document.getElementById("married").value,
        Dependents: document.getElementById("dependents").value,
        Education: document.getElementById("education").value,
        Self_Employed: document.getElementById("selfEmployed").value,
        ApplicantIncome: parseInt(
            document.getElementById("applicantIncome").value
        ),
        CoapplicantIncome: parseInt(
            document.getElementById("coapplicantIncome").value
        ),
        LoanAmount: parseInt(document.getElementById("loanAmount").value),
        Loan_Amount_Term: parseInt(
            document.getElementById("loanAmountTerm").value
        ),
        Credit_History: parseInt(
            document.getElementById("creditHistory").value
        ),
        Property_Area: document.getElementById("propertyArea").value,
    };

    axios
        .post("http://192.168.86.199:5000/predict", formData)
        .then(function (response) {
            let res = response.data.Loan_Status==1 ? "Eligible" : "Not Eligible";
            showModal(response.data.Loan_Status);
        })
        .catch(function (error) {
            console.error("Error:", error);
        });
}

// Function to show modal with message
function showModal(message) {
    var modal = document.getElementById("myModal");
    var modalMessage = document.getElementById("modalMessage");
    modalMessage.innerHTML = "<p>"+message=="1" ? "Congratulations! You are Eligible to get a Loan ✅" :"Sorry! You aren't Eligible to get the Loan ❌" +"</p>";
    modal.style.display = "block";

    // Close the modal when the user clicks anywhere outside of it
    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    };

    // Close the modal when the user clicks on the close button
    var span = document.getElementsByClassName("close")[0];
    span.onclick = function () {
        modal.style.display = "none";
    };
}
