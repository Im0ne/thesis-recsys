$(document).ready(function() {
    const languages = {
        "en": "/static/langs/en.json",
        "cz": "/static/langs/cz.json",
        "ukr": "/static/langs/ukr.json"
    };

    let currentLanguageData;
    let offset = 0;
    const limit = 5;

    const savedLanguage = localStorage.getItem('selectedLanguage') || 'en';
    loadLanguage(savedLanguage);

    function loadLanguage(lang) {
        $.getJSON(languages[lang], function(data) {
            currentLanguageData = data;
            $('#title').text(data.title);
            $('#keywords-label').text(data.keywords_label);
            $('#approach-label').text(data.approach_label);
            $('#submit-button').text(data.submit_button);
            $('#recommendations-heading').text(data.recommendations_heading);
            $('#error-title').text(data.error_title);
            $('#error-message').text(data.error_message);
            $('#goBackBtn').text(data.go_back_button);
            $('#thesis-title').text(data.thesis_title);
            $('#work-type-label').text(data.work_type_label);
            $('label[for="bachelor"]').text(data.bachelor_label);
            $('label[for="diploma"]').text(data.diploma_label);
            $('label[for="dissertation"]').text(data.dissertation_label);
            $('#prev-button').text(data.prev_button);
            $('#next-button').text(data.next_button);
            $('.targets-cz').each(function() {
                $(this).find('strong').text(data.targets_label);
            });
            updateRecommendationsLanguage();
        }).fail(function(jqXHR, textStatus, errorThrown) {
            console.error('Error loading language file:', textStatus, errorThrown); 
        });
    }

    function updateRecommendationsLanguage() {
        $('#recommendations li').each(function() {
            $(this).find('.theme-label').text(currentLanguageData.theme_label);
            $(this).find('.supervisor-label').text(currentLanguageData.supervisor_label);
        });
    }

    function fetchRecommendations(hideSection = true) {
        $('#submit-button').prop('disabled', true);
        $('#recommendations').empty();
        if (hideSection) {
            $('#recommendations-section').hide(); // Hide the recommendations section
        }

        $.ajax({
            type: 'POST',
            url: '/recommendations',
            data: $('#recommendationForm').serialize() + `&offset=${offset}`,
            success: function(data) {
                if (data.error) {
                    $('#recommendations').append(`<li class="list-group-item text-danger">${data.error}</li>`);
                } else if (data.length > 0) {
                    data.forEach(item => {
                        $('#recommendations').append(`
                            <li class="list-group-item clickable-item" style="cursor: pointer;">
                                <div class="recommendation-content">
                                    <div class="d-block text-dark text-decoration-none">
                                        <strong class="theme-label">${currentLanguageData.theme_label}</strong> ${item.Name_CZ} <br>
                                        <strong class="supervisor-label">${currentLanguageData.supervisor_label}</strong> ${item.Supervisor}
                                    </div>
                                    <div class="targets-cz" style="display: none;">
                                        <strong>${currentLanguageData.targets_label}</strong> ${item.Targets_CZ}
                                    </div>
                                </div>
                            </li>
                        `);
                    });
                    updatePaginationButtons(data.length);
                    $('#recommendations-section').show(); // Show the recommendations section
                    updateRecommendationsLanguage(); // Update the language of the recommendations
                } else {
                    $('#recommendations').append(`<li class="list-group-item">${currentLanguageData.no_recommendations}</li>`);
                    $('#recommendations-section').show(); // Show the recommendations section even if no recommendations are found
                }
            },
            error: function(xhr, status, error) {
                console.error('Error fetching recommendations:', xhr.responseJSON.error);
                $('#recommendations').append(`<li class="list-group-item text-danger">Error: ${xhr.responseJSON.error}</li>`);
                $('#recommendations-section').show(); // Show the recommendations section even if there is an error
            },
            complete: function() {
                $('#submit-button').prop('disabled', false);
            }
        });
    }

    function updatePaginationButtons(recommendationsLength) {
        $('#prev-button').prop('disabled', offset === 0);
        $('#next-button').prop('disabled', recommendationsLength < limit);
    }

    $('#recommendationForm').on('submit', function(e) {
        e.preventDefault();
        offset = 0;
        fetchRecommendations(true); 
    });

    $('#prev-button').on('click', function() {
        if (offset > 0) {
            offset -= limit;
            fetchRecommendations(false); 
        }
    });

    $('#next-button').on('click', function() {
        offset += limit;
        fetchRecommendations(false); 
    });

    $('#recommendations').on('click', '.clickable-item', function() {
        $(this).find('.targets-cz').toggle(); 
    });

    const goBackBtn = document.getElementById('goBackBtn');
    if (goBackBtn) {
        goBackBtn.addEventListener('click', function() {
            window.close(); 
            if (!window.closed) {
                window.history.back();
            }
        });
    }

    // Add event listeners to the flag elements
    $('.flag').on('click', function() {
        const selectedLanguage = $(this).data('lang');
        localStorage.setItem('selectedLanguage', selectedLanguage);
        loadLanguage(selectedLanguage);
    });
});