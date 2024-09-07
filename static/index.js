//scroll to top on refresh//
window.onbeforeunload = function() {
  window.scrollTo(0, 0);
}

jQuery(document).ready(function($) {

  //preloader//
  $(window).load(function() {
    $('.loading').delay(2000).fadeOut('slow', function() {
      $(this).remove();
    });
    setTimeout(function() {
      $('.landing').addClass('loaded');
    }, 2000);
    setTimeout(function() {
      $('body').addClass('loaded');
    }, 2000);
    setTimeout(function() {
      $('.landing h2').addClass('loaded');
    }, 3500);
  });

  //change nav bar on scroll//
  window.onscroll = function() {
    if ($(this).scrollTop() > 1) {
      $('header').addClass("resize");
    } else {
      $('header').removeClass("resize");
    }
  };

  //fade in objects on scroll//
  $(window).scroll(function() {
    $('.hideme').each(function(i) {
      var bottom_of_object = $(this).offset().top + $(this).outerHeight();
      var bottom_of_window = $(window).scrollTop() + $(window).height();
      if (bottom_of_window > bottom_of_object) {
        $(this).animate({
          'opacity': '1'
        }, 1250);
      }
    });
  });

  //scroll down on click//
  $(".mouseScroll").click(function() {
    $('html, body').animate({
      scrollTop: $(".about").offset().top - 90
    }, 800);
  });

  //mobile menu//
  window.onload = function() {
    $('.button_container').click(function() {
      $('.button_container').toggleClass('active');
      $('.overlay').toggleClass('open');
      $('body').toggleClass('active');
    });
  }

});