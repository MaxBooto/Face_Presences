$(document).ready(function() {
    console.log("form.js chargé");

    // Fonction pour afficher les messages avec un délai d'effacement
    function showMessage(message, isError = false) {
        console.log(`Affichage message : ${message} (Erreur: ${isError})`);
        $('#message').html(`<div class="alert ${isError ? 'alert-danger' : 'alert-success'}">${message}</div>`);
        setTimeout(function() {
            $('#message').html('');
        }, 5000);
    }

    // Fonction pour gérer l'état des boutons
    function toggleButtons(captureEnabled, trainEnabled, saveEnabled) {
        console.log(`État boutons : capture=${captureEnabled}, train=${trainEnabled}, save=${saveEnabled}`);
        $('#start_capture').prop('disabled', !captureEnabled);
        $('#train_model').prop('disabled', !trainEnabled);
        $('#save_user').prop('disabled', !saveEnabled);
    }

    // Validation des champs
    function validateFields(nom, prenom, sexe) {
        console.log(`Validation champs : nom=${nom}, prenom=${prenom}, sexe=${sexe}`);
        if (!nom || !prenom || !sexe) {
            showMessage("Veuillez remplir tous les champs (nom, prénom, sexe).", true);
            return false;
        }
        return true;
    }

    // Réinitialisation du formulaire
    function resetForm() {
        console.log("Réinitialisation du formulaire");
        $('#person_name').val('');
        $('#person_prenom').val('');
        $('#person_sexe').val('');
        toggleButtons(true, false, false); // Capture activé, autres désactivés
    }

    // Initialiser l'état des boutons
    toggleButtons(true, false, false); // Seul "Capturer" est activé au départ

    $('#start_capture').click(function() {
        console.log("Bouton Capturer les images cliqué");
        var nom = $('#person_name').val().trim();
        var prenom = $('#person_prenom').val().trim();
        var sexe = $('#person_sexe').val();

        if (!validateFields(nom, prenom, sexe)) {
            return;
        }

        toggleButtons(false, false, false); // Désactiver tous les boutons pendant la capture
        $('#message').html('<div class="alert alert-info">Capture en cours...</div>');

        $.ajax({
            url: '/add_user',
            type: 'POST',
            data: {
                action: 'capture',
                nom: nom,
                prenom: prenom,
                sexe: sexe
            },
            success: function(response) {
                console.log("Réponse AJAX (capture):", response);
                if (response.success) {
                    showMessage(response.message);
                    toggleButtons(false, true, false); // Activer "Entraîner le modèle"
                } else {
                    showMessage(response.message, true);
                    toggleButtons(true, false, false); // Revenir à l'état initial
                }
            },
            error: function(xhr) {
                console.log("Erreur AJAX (capture):", xhr);
                var errorMsg = xhr.responseJSON && xhr.responseJSON.message ? xhr.responseJSON.message : 'Erreur lors de la capture.';
                showMessage(errorMsg, true);
                toggleButtons(true, false, false);
            }
        });
    });

    $('#train_model').click(function() {
        console.log("Bouton Entraîner le modèle cliqué");
        toggleButtons(false, false, false); // Désactiver tous les boutons pendant l'entraînement
        $('#message').html('<div class="alert alert-info">Entraînement en cours...</div>');

        $.ajax({
            url: '/train',
            type: 'POST',
            success: function(response) {
                console.log("Réponse AJAX (train):", response);
                if (response.success) {
                    showMessage(response.message);
                    toggleButtons(false, false, true); // Activer "Enregistrer"
                    console.log("Bouton Enregistrer activé après entraînement réussi");
                } else {
                    showMessage(response.message, true); // Afficher le message d'erreur exact
                    toggleButtons(false, true, false); // Revenir à "Entraîner" activé
                }
            },
            error: function(xhr) {
                console.log("Erreur AJAX (train):", xhr);
                var errorMsg = xhr.responseJSON && xhr.responseJSON.message ? xhr.responseJSON.message : 'Erreur lors de l\'entraînement.';
                showMessage(errorMsg, true);
                toggleButtons(false, true, false);
            }
        });
    });

    $('#save_user').click(function() {
        console.log("Bouton Enregistrer cliqué");
        var nom = $('#person_name').val().trim();
        var prenom = $('#person_prenom').val().trim();
        var sexe = $('#person_sexe').val();

        if (!validateFields(nom, prenom, sexe)) {
            return;
        }

        toggleButtons(false, false, false); // Désactiver tous les boutons pendant l'enregistrement
        $('#message').html('<div class="alert alert-info">Enregistrement en cours...</div>');

        $.ajax({
            url: '/add_user',
            type: 'POST',
            data: {
                action: 'save',
                nom: nom,
                prenom: prenom,
                sexe: sexe
            },
            success: function(response) {
                console.log("Réponse AJAX (save):", response);
                if (response.success) {
                    showMessage(response.message);
                    setTimeout(function() {
                        console.log("Redirection vers /manage_users après enregistrement");
                        window.location.href = '/manage_users';
                    }, 1000);
                } else {
                    showMessage(response.message, true);
                    toggleButtons(false, false, true); // Revenir à "Enregistrer" activé
                }
            },
            error: function(xhr) {
                console.log("Erreur AJAX (save):", xhr);
                var errorMsg = xhr.responseJSON && xhr.responseJSON.message ? xhr.responseJSON.message : 'Erreur lors de l\'enregistrement.';
                showMessage(errorMsg, true);
                toggleButtons(false, false, true);
            }
        });
    });
});