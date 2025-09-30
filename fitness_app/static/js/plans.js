// Plans and dashboard functionality for FitPlan

document.addEventListener('DOMContentLoaded', function() {
    initializeTabs();
    initializeGroceryList();
    initializeWorkoutTracking();
    initializeMealLogging();
});

// Tab switching functionality
function initializeTabs() {
    const tabOptions = document.querySelectorAll('.tab-option');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabOptions.forEach(tab => {
        tab.addEventListener('click', function() {
            const targetTab = this.dataset.tab;
            
            // Remove active class from all tabs and content
            tabOptions.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Show corresponding content
            const targetContent = document.getElementById(`${targetTab}-tab`);
            if (targetContent) {
                targetContent.classList.add('active');
            }
            
            // Smooth scroll to top of content
            const scrollableContent = document.querySelector('.scrollable-content');
            if (scrollableContent) {
                scrollableContent.scrollTop = 0;
            }
            
            // Track tab usage (could be sent to analytics)
            trackTabUsage(targetTab);
        });
    });
}

// Grocery list interactions
function initializeGroceryList() {
    const groceryItems = document.querySelectorAll('.grocery-item');
    
    groceryItems.forEach(item => {
        item.addEventListener('click', function() {
            this.classList.toggle('checked');
            updateGroceryProgress();
            
            // Save checked state to localStorage
            const itemName = this.querySelector('.grocery-item-name').textContent;
            saveGroceryItemState(itemName, this.classList.contains('checked'));
        });
    });
    
    // Load saved grocery states
    loadGroceryStates();
}

function updateGroceryProgress() {
    const allItems = document.querySelectorAll('.grocery-item');
    const checkedItems = document.querySelectorAll('.grocery-item.checked');
    const totalItems = allItems.length;
    const checkedCount = checkedItems.length;
    
    // Update any progress indicators
    const progressText = document.querySelector('.summary-row');
    if (progressText && progressText.textContent.includes('Items checked')) {
        progressText.innerHTML = `
            <span>Items checked:</span>
            <span>${checkedCount} of ${totalItems}</span>
        `;
    }
    
    // Show completion message
    if (checkedCount === totalItems && totalItems > 0) {
        if (window.FitPlan) {
            window.FitPlan.showSuccess('Shopping list completed! 🎉');
        }
    }
}

function saveGroceryItemState(itemName, isChecked) {
    const storageKey = 'fitplan_grocery_items';
    const existing = JSON.parse(localStorage.getItem(storageKey) || '{}');
    existing[itemName] = isChecked;
    localStorage.setItem(storageKey, JSON.stringify(existing));
}

function loadGroceryStates() {
    const storageKey = 'fitplan_grocery_items';
    const savedStates = JSON.parse(localStorage.getItem(storageKey) || '{}');
    
    Object.keys(savedStates).forEach(itemName => {
        const item = [...document.querySelectorAll('.grocery-item-name')]
            .find(el => el.textContent === itemName)?.closest('.grocery-item');
        
        if (item && savedStates[itemName]) {
            item.classList.add('checked');
        }
    });
    
    updateGroceryProgress();
}

// Workout tracking
function initializeWorkoutTracking() {
    const workoutDays = document.querySelectorAll('.workout-day:not(.rest)');
    
    workoutDays.forEach(workoutDay => {
        // Add click handler for workout completion
        const header = workoutDay.querySelector('.workout-day-header');
        if (header) {
            const completeBtn = document.createElement('button');
            completeBtn.className = 'workout-complete-btn';
            completeBtn.innerHTML = '✓';
            completeBtn.title = 'Mark as complete';
            completeBtn.style.cssText = `
                background: #52c41a;
                color: white;
                border: none;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                font-size: 12px;
                cursor: pointer;
                margin-left: 8px;
            `;
            
            header.appendChild(completeBtn);
            
            completeBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                markWorkoutComplete(workoutDay);
            });
        }
        
        // Load saved workout states
        const dayTitle = workoutDay.querySelector('.workout-day-title').textContent;
        if (isWorkoutCompleted(dayTitle)) {
            markWorkoutComplete(workoutDay, false); // Don't show message
        }
    });
}

function markWorkoutComplete(workoutDay, showMessage = true) {
    workoutDay.classList.add('completed');
    workoutDay.style.opacity = '0.7';
    
    const completeBtn = workoutDay.querySelector('.workout-complete-btn');
    if (completeBtn) {
        completeBtn.style.background = '#52c41a';
        completeBtn.innerHTML = '✓';
    }
    
    // Save completion state
    const dayTitle = workoutDay.querySelector('.workout-day-title').textContent;
    saveWorkoutCompletion(dayTitle);
    
    if (showMessage && window.FitPlan) {
        window.FitPlan.showSuccess('Workout completed! Great job! 💪');
    }
    
    updateWorkoutProgress();
}

function saveWorkoutCompletion(dayTitle) {
    const storageKey = 'fitplan_completed_workouts';
    const existing = JSON.parse(localStorage.getItem(storageKey) || '[]');
    if (!existing.includes(dayTitle)) {
        existing.push(dayTitle);
        localStorage.setItem(storageKey, JSON.stringify(existing));
    }
}

function isWorkoutCompleted(dayTitle) {
    const storageKey = 'fitplan_completed_workouts';
    const completed = JSON.parse(localStorage.getItem(storageKey) || '[]');
    return completed.includes(dayTitle);
}

function updateWorkoutProgress() {
    const totalWorkouts = document.querySelectorAll('.workout-day:not(.rest)').length;
    const completedWorkouts = document.querySelectorAll('.workout-day.completed').length;
    
    // You could update a progress indicator here
    console.log(`Workout progress: ${completedWorkouts}/${totalWorkouts}`);
}

// Meal logging
function initializeMealLogging() {
    const mealCards = document.querySelectorAll('.meal-card');
    
    mealCards.forEach(mealCard => {
        const header = mealCard.querySelector('.meal-header');
        if (header) {
            const logBtn = document.createElement('button');
            logBtn.className = 'meal-log-btn';
            logBtn.innerHTML = 'Log';
            logBtn.title = 'Log this meal';
            logBtn.style.cssText = `
                background: #667eea;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 10px;
                cursor: pointer;
                margin-left: 8px;
            `;
            
            header.appendChild(logBtn);
            
            logBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                logMeal(mealCard);
            });
        }
        
        // Load saved meal states
        const mealTitle = mealCard.querySelector('.meal-title').textContent;
        if (isMealLogged(mealTitle)) {
            logMeal(mealCard, false); // Don't show message
        }
    });
}

function logMeal(mealCard, showMessage = true) {
    mealCard.classList.add('logged');
    
    const logBtn = mealCard.querySelector('.meal-log-btn');
    if (logBtn) {
        logBtn.style.background = '#52c41a';
        logBtn.innerHTML = '✓';
        logBtn.title = 'Meal logged';
    }
    
    // Save logged state
    const mealTitle = mealCard.querySelector('.meal-title').textContent;
    saveMealLog(mealTitle);
    
    if (showMessage && window.FitPlan) {
        window.FitPlan.showSuccess('Meal logged successfully! 🥗');
    }
    
    updateMealProgress();
}

function saveMealLog(mealTitle) {
    const storageKey = 'fitplan_logged_meals';
    const existing = JSON.parse(localStorage.getItem(storageKey) || '[]');
    if (!existing.includes(mealTitle)) {
        existing.push(mealTitle);
        localStorage.setItem(storageKey, JSON.stringify(existing));
    }
}

function isMealLogged(mealTitle) {
    const storageKey = 'fitplan_logged_meals';
    const logged = JSON.parse(localStorage.getItem(storageKey) || '[]');
    return logged.includes(mealTitle);
}

function updateMealProgress() {
    const totalMeals = document.querySelectorAll('.meal-card').length;
    const loggedMeals = document.querySelectorAll('.meal-card.logged').length;
    
    // You could update a progress indicator here
    console.log(`Meal progress: ${loggedMeals}/${totalMeals}`);
}

// Analytics/tracking functions
function trackTabUsage(tabName) {
    // This would send data to your analytics service
    console.log(`Tab viewed: ${tabName}`);
    
    // Example: Send to Google Analytics, Mixpanel, etc.
    // analytics.track('Tab Viewed', { tab: tabName });
}

// Export functions for external use
window.PlansApp = {
    markWorkoutComplete,
    logMeal,
    updateGroceryProgress,
    trackTabUsage
};

// Add CSS for completed states
const style = document.createElement('style');
style.textContent = `
    .workout-day.completed {
        opacity: 0.7;
        position: relative;
    }
    
    .workout-day.completed::after {
        content: '✓ Completed';
        position: absolute;
        top: 10px;
        right: 10px;
        background: #52c41a;
        color: white;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: bold;
    }
    
    .meal-card.logged {
        position: relative;
    }
    
    .meal-card.logged::after {
        content: '✓ Logged';
        position: absolute;
        top: 10px;
        right: 10px;
        background: #52c41a;
        color: white;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: bold;
    }
    
    .workout-complete-btn:hover {
        transform: scale(1.1);
        transition: transform 0.2s ease;
    }
    
    .meal-log-btn:hover {
        transform: scale(1.05);
        transition: transform 0.2s ease;
    }
`;
document.head.appendChild(style);