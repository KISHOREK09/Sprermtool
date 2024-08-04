document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const loaderWrapper = document.getElementById('loader-wrapper');
    const messageDiv = document.getElementById('message');
    const greetingP = document.getElementById('greeting');
    const usernameInput = document.getElementById('username');
    const imageForm = document.getElementById('imageForm');
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const alertDiv = document.querySelector('.alert');
    const resultDiv = document.querySelector('.result');
    const resultSection = document.getElementById('resultSection');

    if (loginForm) {
        loginForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const username = usernameInput.value.trim();
            if (username) {
                showLoader();

                setTimeout(() => {
                    loaderWrapper.style.display = 'none';
                    greetingP.innerText = `Hello, ${username}! Please upload your image.`;
                    messageDiv.style.display = 'block';
                    loginForm.style.display = 'none';
                    step1.style.display = 'none';
                    step2.style.display = 'block';
                }, 2000);
            }
        });
    }

    if (imageForm) {
        imageForm.addEventListener('submit', function(event) {
            showLoader();
            // Allow the form to submit
        });
    }

    function showLoader() {
        loaderWrapper.style.display = 'flex';
    }

    window.resetForm = function() {
        usernameInput.value = '';
        greetingP.innerText = '';
        messageDiv.style.display = 'none';
        loginForm.style.display = 'block';
        step1.style.display = 'block';
        step2.style.display = 'none';
        if (alertDiv) alertDiv.style.display = 'none';
        if (resultDiv) resultDiv.style.display = 'none';
    }

    function showResult() {
        if (resultSection) {
            messageDiv.style.display = 'none';
            step1.style.display = 'none';
            step2.style.display = 'none';
            resultSection.style.display = 'block';
        }
    }

    if (resultSection && resultSection.style.display === 'block') {
        showResult();
    }
});
