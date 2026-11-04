import { gsap } from "gsap";

export function attachLandingMotion(root: HTMLElement) {
  const disposers: (() => void)[] = [];
  const media = gsap.matchMedia();
  media.add("(prefers-reduced-motion: no-preference)", () => {
    const context = gsap.context(() => {
      gsap.from("[data-entrance]", {
        y: 16,
        opacity: 0.65,
        duration: 1.1,
        stagger: 0.13,
        ease: "power2.out",
        clearProps: "all",
      });
    }, root);
    root.querySelectorAll<HTMLElement>("[data-micro]").forEach((element) => {
      const cta = element.dataset.micro === "cta";
      const animate = (scale: number, y: number) =>
        gsap.to(element, {
          scale,
          y,
          duration: 0.25,
          overwrite: true,
          ease: "power2.out",
        });
      const enter = () => animate(cta ? 1.035 : 1.03, -1);
      const leave = () => animate(1, 0);
      const down = () => animate(0.97, 0);
      const events: [string, EventListener][] = [
        ["pointerenter", enter],
        ["pointerleave", leave],
        ["focus", enter],
        ["blur", leave],
        ["pointerdown", down],
        ["pointerup", enter],
      ];
      events.forEach(([name, handler]) =>
        element.addEventListener(name, handler),
      );
      disposers.push(() => {
        events.forEach(([name, handler]) =>
          element.removeEventListener(name, handler),
        );
        gsap.killTweensOf(element);
        gsap.set(element, { clearProps: "transform" });
      });
    });
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          context.add(() =>
            gsap.from(entry.target, {
              y: 20,
              opacity: 0.45,
              duration: 0.8,
              clearProps: "all",
            }),
          );
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.15 },
    );
    root
      .querySelectorAll("[data-reveal]")
      .forEach((element) => observer.observe(element));
    return () => {
      observer.disconnect();
      context.revert();
      disposers.splice(0).forEach((dispose) => dispose());
    };
  });
  return () => media.revert();
}
