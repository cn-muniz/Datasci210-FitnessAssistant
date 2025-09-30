// Main JavaScript functionality for FitPlan app

document.addEventListener('DOMContentLoaded', function() {
    // Flash message auto-hide
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(message => {
        setTimeout(() => {
            message.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => {
                message.remove();
            }, 300);
        }, 5000);
    });

    // Add slideOut animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideOut {
            from {
                transform: translate(-50%, 0);
                opacity: 1;
            }
            to {
                transform: translate(-50%, -100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);

    // Global form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const requiredFields = form.querySelectorAll('[required]');
            let isValid = true;

            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    field.classList.add('error');
                    isValid = false;
                } else {
                    field.classList.remove('error');
                }
            });

            if (!isValid) {
                e.preventDefault();
                showError('Please fill in all required fields');
            }
        });
    });

    // Remove error class on input
    const inputs = document.querySelectorAll('.form-input');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.classList.contains('error') && this.value.trim()) {
                this.classList.remove('error');
            }
        });
    });
});

// Utility functions
function showError(message) {
    const flashContainer = document.querySelector('.flash-messages') || createFlashContainer();
    const flashMessage = document.createElement('div');
    flashMessage.className = 'flash-message flash-error';
    flashMessage.textContent = message;
    flashContainer.appendChild(flashMessage);

    // Auto-hide after 5 seconds
    setTimeout(() => {
        flashMessage.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => {
            flashMessage.remove();
        }, 300);
    }, 5000);
}

function showSuccess(message) {
    const flashContainer = document.querySelector('.flash-messages') || createFlashContainer();
    const flashMessage = document.createElement('div');
    flashMessage.className = 'flash-message';
    flashMessage.style.background = '#d4edda';
    flashMessage.style.color = '#155724';
    flashMessage.textContent = message;
    flashContainer.appendChild(flashMessage);

    // Auto-hide after 3 seconds
    setTimeout(() => {
        flashMessage.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => {
            flashMessage.remove();
        }, 300);
    }, 3000);
}

function createFlashContainer() {
    const container = document.createElement('div');
    container.className = 'flash-messages';
    document.body.appendChild(container);
    return container;
}

// Loading state for buttons
function setButtonLoading(button, isLoading = true) {
    if (isLoading) {
        button.classList.add('btn-loading');
        button.disabled = true;
        button.dataset.originalText = button.textContent;
        button.textContent = '';
    } else {
        button.classList.remove('btn-loading');
        button.disabled = false;
        button.textContent = button.dataset.originalText || button.textContent;
    }
}

// Smooth scrolling for mobile
function smoothScrollTo(element) {
    if (element) {
        element.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Local storage helpers (for temporary data)
function saveFormProgress(step, data) {
    const progressKey = 'fitplan_progress';
    const existing = JSON.parse(localStorage.getItem(progressKey) || '{}');
    existing[step] = data;
    localStorage.setItem(progressKey, JSON.stringify(existing));
}

function getFormProgress(step) {
    const progressKey = 'fitplan_progress';
    const existing = JSON.parse(localStorage.getItem(progressKey) || '{}');
    return existing[step] || {};
}

function clearFormProgress() {
    localStorage.removeItem('fitplan_progress');
}

// Export functions for use in other files
window.FitPlan = {
    showError,
    showSuccess,
    setButtonLoading,
    smoothScrollTo,
    saveFormProgress,
    getFormProgress,
    clearFormProgress
};